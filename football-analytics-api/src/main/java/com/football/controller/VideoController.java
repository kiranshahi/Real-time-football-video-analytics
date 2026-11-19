package com.football.controller;

import com.football.service.VideoInferenceService;
import org.springframework.http.*;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import java.nio.file.*;

@RestController
@RequestMapping("/api/video")
public class VideoController {

    private final VideoInferenceService service;

    public VideoController(VideoInferenceService service) {
        this.service = service;
    }

    @PostMapping(value = "/predict",  consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<byte[]> predict(@RequestParam("file") MultipartFile file) throws Exception {

        Path input = Files.createTempFile("input-", ".mp4");
        Path output = Files.createTempFile("output-", ".mp4");

        try {
            file.transferTo(input);

            service.process(input.toString(),  output.toString());

            byte[] video = Files.readAllBytes(output);

            return ResponseEntity.ok()
                    .contentType(MediaType.parseMediaType("video/mp4"))
                    .header(HttpHeaders.CONTENT_DISPOSITION,  "attachment; filename=detected.mp4")
                    .body(video);

        } finally {
            Files.deleteIfExists(input);
            Files.deleteIfExists(output);
        }
    }
}