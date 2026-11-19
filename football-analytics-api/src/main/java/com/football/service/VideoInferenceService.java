package com.football.service;

import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.FFmpegFrameRecorder;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.Frame;

import org.bytedeco.ffmpeg.global.avcodec;
import org.springframework.stereotype.Service;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.BasicStroke;
import java.awt.Font;
import java.awt.image.BufferedImage;

import java.nio.file.Path;
import java.util.List;

@Service
public class VideoInferenceService {
    private final YoloInferenceService yolo;

    public VideoInferenceService(YoloInferenceService yolo) {
        this.yolo = yolo;
    }

    public void process(String inputPath, String outputPath)
            throws Exception {

        try (FFmpegFrameGrabber grabber =  new FFmpegFrameGrabber(inputPath);
             Java2DFrameConverter converter =  new Java2DFrameConverter()) {

            grabber.start();

            int width = grabber.getImageWidth();
            int height = grabber.getImageHeight();
            double fps = grabber.getVideoFrameRate();

            if (width <= 0 || height <= 0) {
                throw new IllegalArgumentException("Input contains no video stream");
            }

            if (!Double.isFinite(fps) || fps <= 0) {
                fps = 25.0;
            }

            try (FFmpegFrameRecorder recorder = new FFmpegFrameRecorder(outputPath, width, height)) {

                recorder.setFormat("mp4");
                recorder.setVideoCodec(avcodec.AV_CODEC_ID_H264);
                recorder.setFrameRate(fps);
                recorder.setVideoBitrate(4_000_000);
                recorder.start();

                int frameNumber = 0;
                Frame frame;

                while ((frame = grabber.grabImage()) != null) {

                    BufferedImage image = converter.convert(frame);

                    if (image == null) continue;

                    List<YoloInferenceService.Detection>  detections = yolo.predict(image);

                    drawDetections(image, detections);

                    recorder.record(converter.convert(image));

                    frameNumber++;

                    if (frameNumber % 30 == 0) {
                        System.out.printf("Processed %d frames%n", frameNumber);
                    }
                }

                System.out.println("Video saved: " + Path.of(outputPath).toAbsolutePath()
                );
            }
        }
    }

    private void drawDetections(BufferedImage image, List<YoloInferenceService.Detection> detections) {

        Graphics2D g = image.createGraphics();

        try {
            g.setStroke(new BasicStroke(3));
            g.setFont(new Font("Arial", Font.BOLD, 18));

            for (var d : detections) {

                Color color = switch (d.label()) {
                    case "player" -> Color.GREEN;
                    case "referee" -> Color.YELLOW;
                    case "goalkeeper" -> Color.RED;
                    default -> Color.WHITE;
                };

                g.setColor(color);

                int x = Math.round(d.x1());
                int y = Math.round(d.y1());
                int w = Math.round(d.x2() - d.x1());
                int h = Math.round(d.y2() - d.y1());

                g.drawRect(x, y, w, h);

                String label = String.format("%s %.2f", d.label(), d.confidence());

                g.drawString(label, x, Math.max(20, y - 5));
            }
        } finally {
            g.dispose();
        }
    }
}
