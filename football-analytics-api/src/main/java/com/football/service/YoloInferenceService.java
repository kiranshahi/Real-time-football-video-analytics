
package com.football.service;

import ai.onnxruntime.*;
import jakarta.annotation.PreDestroy;
import org.springframework.stereotype.Service;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

@Service
public class YoloInferenceService {

    private static final int IMAGE_SIZE = 960;
    private static final float CONF_THRESHOLD = 0.25f;
    private static final float IOU_THRESHOLD = 0.45f;

    // Check that this matches your training dataset.yaml
    private static final String[] CLASS_NAMES = {
            "player",
            "referee",
            "goalkeeper"
    };

    private final OrtEnvironment environment;
    private final OrtSession session;
    private final String inputName;

    public record Detection(
            String label,
            float confidence,
            float x1,
            float y1,
            float x2,
            float y2
    ) {}

    public YoloInferenceService() throws OrtException {

        environment = OrtEnvironment.getEnvironment();

        String modelPath = Path.of(
                "models", "best.onnx"
        ).toAbsolutePath().toString();

        OrtSession.SessionOptions options =
                new OrtSession.SessionOptions();

        try {
            session = environment.createSession(
                    modelPath, options
            );
        } finally {
            options.close();
        }

        inputName = session.getInputNames()
                .iterator().next();

        System.out.println("YOLO model loaded: " + modelPath);
        System.out.println("Inputs: " + session.getInputInfo());
        System.out.println("Outputs: " + session.getOutputInfo());
    }

    public List<Detection> predict(BufferedImage image)
            throws OrtException {

        int originalWidth = image.getWidth();
        int originalHeight = image.getHeight();

        // 1. Letterbox resize while preserving aspect ratio
        float scale = Math.min(
                (float) IMAGE_SIZE / originalWidth,
                (float) IMAGE_SIZE / originalHeight
        );

        int resizedWidth = Math.round(originalWidth * scale);
        int resizedHeight = Math.round(originalHeight * scale);

        int padX = (IMAGE_SIZE - resizedWidth) / 2;
        int padY = (IMAGE_SIZE - resizedHeight) / 2;

        BufferedImage resized = new BufferedImage(
                IMAGE_SIZE,
                IMAGE_SIZE,
                BufferedImage.TYPE_INT_RGB
        );

        Graphics2D graphics = resized.createGraphics();

        try {
            graphics.setColor(new Color(114, 114, 114));
            graphics.fillRect(
                    0, 0, IMAGE_SIZE, IMAGE_SIZE
            );

            graphics.setRenderingHint(
                    RenderingHints.KEY_INTERPOLATION,
                    RenderingHints.VALUE_INTERPOLATION_BILINEAR
            );

            graphics.drawImage(
                    image,
                    padX, padY,
                    resizedWidth, resizedHeight,
                    null
            );

        } finally {
            graphics.dispose();
        }

        // 2. Convert image to NCHW RGB float32 tensor
        float[][][][] input =
                new float[1][3][IMAGE_SIZE][IMAGE_SIZE];

        for (int y = 0; y < IMAGE_SIZE; y++) {
            for (int x = 0; x < IMAGE_SIZE; x++) {

                int rgb = resized.getRGB(x, y);

                input[0][0][y][x] =
                        ((rgb >> 16) & 0xFF) / 255.0f;

                input[0][1][y][x] =
                        ((rgb >> 8) & 0xFF) / 255.0f;

                input[0][2][y][x] =
                        (rgb & 0xFF) / 255.0f;
            }
        }

        // 3. Run ONNX inference
        try (
                OnnxTensor inputTensor = OnnxTensor.createTensor(environment, input);

                OrtSession.Result results = session.run(Map.of(inputName, inputTensor))
        ) {

            float[][][] output = (float[][][]) results.get(0).getValue();

            // 4. Decode predictions and perform NMS
            return postprocess(output[0], scale, padX, padY, originalWidth, originalHeight);
        }
    }

    private List<Detection> postprocess(
            float[][] output,
            float scale,
            int padX,
            int padY,
            int imageWidth,
            int imageHeight
    ) {

        int attributes = 4 + CLASS_NAMES.length;

        // Supports [attributes, boxes] and [boxes, attributes].
        boolean transposed = output.length == attributes;

        if (!transposed && output[0].length != attributes) {
            throw new IllegalStateException("Unexpected YOLO output dimensions: " +  output.length + " x " + output[0].length);
        }

        int numBoxes = transposed ? output[0].length : output.length;

        List<Detection> candidates = new ArrayList<>();

        for (int i = 0; i < numBoxes; i++) {

            int bestClass = -1;
            float bestConfidence = 0.0f;

            for (int c = 0; c < CLASS_NAMES.length; c++) {

                float confidence = transposed
                        ? output[c + 4][i]
                        : output[i][c + 4];

                if (confidence > bestConfidence) {
                    bestConfidence = confidence;
                    bestClass = c;
                }
            }

            if (bestClass < 0 || bestConfidence < CONF_THRESHOLD) {
                continue;
            }

            // YOLO: center-x, center-y, width, height
            float cx = transposed
                    ? output[0][i] : output[i][0];
            float cy = transposed
                    ? output[1][i] : output[i][1];
            float w = transposed
                    ? output[2][i] : output[i][2];
            float h = transposed
                    ? output[3][i] : output[i][3];

            // Convert back to original image coordinates
            float x1 = clamp(
                    (cx - w / 2 - padX) / scale,
                    0, imageWidth
            );

            float y1 = clamp(
                    (cy - h / 2 - padY) / scale,
                    0, imageHeight
            );

            float x2 = clamp(
                    (cx + w / 2 - padX) / scale,
                    0, imageWidth
            );

            float y2 = clamp(
                    (cy + h / 2 - padY) / scale,
                    0, imageHeight
            );

            if (x2 <= x1 || y2 <= y1) continue;

            candidates.add(new Detection(
                    CLASS_NAMES[bestClass],
                    bestConfidence,
                    x1, y1, x2, y2
            ));
        }

        // 5. Non-Maximum Suppression
        candidates.sort(
                Comparator.comparingDouble(
                        Detection::confidence
                ).reversed()
        );

        List<Detection> finalDetections =
                new ArrayList<>();

        for (Detection candidate : candidates) {

            boolean suppress = false;

            for (Detection selected : finalDetections) {

                if (candidate.label().equals(selected.label())
                        && calculateIoU(candidate, selected)
                        > IOU_THRESHOLD) {

                    suppress = true;
                    break;
                }
            }

            if (!suppress) {
                finalDetections.add(candidate);
            }
        }

        return finalDetections;
    }

    private float calculateIoU(Detection a, Detection b) {

        float x1 = Math.max(a.x1(), b.x1());
        float y1 = Math.max(a.y1(), b.y1());
        float x2 = Math.min(a.x2(), b.x2());
        float y2 = Math.min(a.y2(), b.y2());

        float intersection =
                Math.max(0, x2 - x1) *
                        Math.max(0, y2 - y1);

        float areaA =
                (a.x2() - a.x1()) * (a.y2() - a.y1());

        float areaB =
                (b.x2() - b.x1()) * (b.y2() - b.y1());

        return intersection /
                Math.max(1e-6f, areaA + areaB - intersection);
    }

    private float clamp(
            float value, float min, float max
    ) {
        return Math.max(min, Math.min(max, value));
    }

    @PreDestroy
    public void close() throws OrtException {
        session.close();
    }
}
