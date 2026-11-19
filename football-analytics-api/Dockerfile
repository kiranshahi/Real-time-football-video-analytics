
# Build Spring Boot application
FROM maven:3.9-eclipse-temurin-21 AS build

WORKDIR /app

COPY pom.xml .
RUN mvn dependency:go-offline -B

COPY src ./src

RUN mvn clean package -DskipTests

# Run Java application
FROM eclipse-temurin:21-jre-jammy

WORKDIR /app

# Install FFmpeg runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg fontconfig && \
    rm -rf /var/lib/apt/lists/*

# Copy Spring Boot application
COPY --from=build /app/target/*.jar app.jar

# Copy trained YOLO model
COPY models ./models

# Copy sample videos
COPY videos ./videos

EXPOSE 8080

ENV JAVA_TOOL_OPTIONS="-Xmx2g"

ENTRYPOINT ["java", "-jar", "app.jar"]
