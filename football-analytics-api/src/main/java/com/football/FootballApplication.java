package com.football;

import com.football.service.VideoInferenceService;
import org.springframework.boot.*;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;

@SpringBootApplication
public class FootballApplication {
    public static void main(String[] args) {
        SpringApplication.run(
                FootballApplication.class, args
        );
    }
}
