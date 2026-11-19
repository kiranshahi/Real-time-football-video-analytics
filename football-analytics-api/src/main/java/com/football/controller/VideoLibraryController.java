
package com.football.controller;

import com.football.service.YoloInferenceService;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.Frame;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;
import org.springframework.http.*;
import org.springframework.web.bind.annotation.*;

import java.awt.image.BufferedImage;
import java.nio.file.*;
import java.util.List;

@RestController
@RequestMapping("/api/videos")
public class VideoLibraryController {

    private final YoloInferenceService yolo;

    private static final Path VIDEO_DIR =
            Path.of("videos").toAbsolutePath().normalize();

    // Only these videos are available to users
    private static final List<String> ALLOWED_VIDEOS = List.of("test1.mp4", "test2.mp4", "test3.mp4");

    public VideoLibraryController(YoloInferenceService yolo) {
        this.yolo = yolo;
    }

    public record VideoItem(String name, String url) {}

    private Path resolveVideo(String name) {
        if (!ALLOWED_VIDEOS.contains(name)) {
            throw new org.springframework.web.server.ResponseStatusException(HttpStatus.NOT_FOUND, "Unknown video");
        }

        Path path = VIDEO_DIR.resolve(name).normalize();

        if (!path.startsWith(VIDEO_DIR) || !Files.isRegularFile(path) ||  !Files.isReadable(path)) {

            throw new org.springframework.web.server.ResponseStatusException(HttpStatus.NOT_FOUND, "Video not found");
        }

        return path;
    }

    // List available server videos
    @GetMapping
    public List<VideoItem> listVideos() {
        return ALLOWED_VIDEOS.stream()
                .filter(name -> {
                    Path path = VIDEO_DIR.resolve(name);
                    return Files.isRegularFile(path)
                            && Files.isReadable(path);
                })
                .map(name -> new VideoItem(
                        name,
                        "/api/videos/" + name + "/stream"
                ))
                .toList();
    }

    // Serve video to browser (supports HTTP ranges)
    @GetMapping("/{name}/stream")
    public ResponseEntity<Resource> streamVideo(
            @PathVariable String name) {

        Path path = resolveVideo(name);
        Resource resource = new FileSystemResource(path);

        return ResponseEntity.ok()
                .contentType(MediaType.parseMediaType("video/mp4"))
                .body(resource);
    }

    // Run YOLO inference on a frame at selected time
    @GetMapping("/{name}/detect")
    public List<YoloInferenceService.Detection> detect(
            @PathVariable String name,
            @RequestParam double time) throws Exception {

        Path path = resolveVideo(name);

        if (!Double.isFinite(time) || time < 0) {
            throw new org.springframework.web.server
                    .ResponseStatusException(
                    HttpStatus.BAD_REQUEST, "Invalid time"
            );
        }

        try (FFmpegFrameGrabber grabber =
                     new FFmpegFrameGrabber(path.toFile());
             Java2DFrameConverter converter =
                     new Java2DFrameConverter()) {

            grabber.start();

            long timestampUs = Math.round(time * 1_000_000);
            long durationUs = grabber.getLengthInTime();

            if (durationUs > 0 && timestampUs >= durationUs) {
                throw new org.springframework.web.server
                        .ResponseStatusException(
                        HttpStatus.BAD_REQUEST,
                        "Time exceeds video duration"
                );
            }

            grabber.setTimestamp(timestampUs);

            Frame frame = grabber.grabImage();

            if (frame == null) {
                return List.of();
            }

            BufferedImage image = converter.convert(frame);

            if (image == null) {
                return List.of();
            }

            return yolo.predict(image);
        }
    }
}
