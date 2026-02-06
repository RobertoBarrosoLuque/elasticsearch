/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.services.fireworksai;

import org.elasticsearch.common.settings.SecureString;
import org.elasticsearch.core.Nullable;
import org.elasticsearch.inference.ChunkingSettings;
import org.elasticsearch.inference.EmptyTaskSettings;
import org.elasticsearch.inference.Model;
import org.elasticsearch.inference.SimilarityMeasure;
import org.elasticsearch.inference.TaskType;
import org.elasticsearch.threadpool.ThreadPool;
import org.elasticsearch.xpack.inference.external.http.HttpClientManager;
import org.elasticsearch.xpack.inference.external.http.sender.HttpRequestSenderTests;
import org.elasticsearch.xpack.inference.services.AbstractInferenceServiceTests;
import org.elasticsearch.xpack.inference.services.ConfigurationParseContext;
import org.elasticsearch.xpack.inference.services.SenderService;
import org.elasticsearch.xpack.inference.services.ServiceFields;
import org.elasticsearch.xpack.inference.services.fireworksai.embeddings.FireworksAiEmbeddingsModel;
import org.elasticsearch.xpack.inference.services.fireworksai.embeddings.FireworksAiEmbeddingsServiceSettings;
import org.elasticsearch.xpack.inference.services.settings.DefaultSecretSettings;

import java.net.URI;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.Map;

import static org.elasticsearch.xpack.inference.Utils.mockClusterServiceEmpty;
import static org.elasticsearch.xpack.inference.services.ServiceComponentsTests.createWithEmptySettings;
import static org.elasticsearch.xpack.inference.services.settings.DefaultSecretSettingsTests.getSecretSettingsMap;
import static org.hamcrest.Matchers.instanceOf;
import static org.hamcrest.core.Is.is;

public class FireworksAiServiceTests extends AbstractInferenceServiceTests {

    private static final String MODEL = "model";
    private static final SimilarityMeasure SIMILARITY = SimilarityMeasure.DOT_PRODUCT;
    private static final int DIMENSIONS = 100;
    private static final String SECRET = "secret";
    private static final String INFERENCE_ID = "id";
    private static final String DEFAULT_URL = "https://api.fireworks.ai/inference/v1/embeddings";

    public FireworksAiServiceTests() {
        super(createTestConfiguration());
    }

    private static TestConfiguration createTestConfiguration() {
        return new TestConfiguration.Builder(
            new CommonConfig(TaskType.TEXT_EMBEDDING, TaskType.COMPLETION, EnumSet.of(TaskType.TEXT_EMBEDDING)) {
                @Override
                protected SenderService createService(ThreadPool threadPool, HttpClientManager clientManager) {
                    return FireworksAiServiceTests.createService(threadPool, clientManager);
                }

                @Override
                protected Map<String, Object> createServiceSettingsMap(TaskType taskType) {
                    return createServiceSettingsMap(taskType, ConfigurationParseContext.REQUEST);
                }

                @Override
                protected Map<String, Object> createServiceSettingsMap(TaskType taskType, ConfigurationParseContext parseContext) {
                    return FireworksAiServiceTests.createServiceSettingsMap(taskType, parseContext);
                }

                @Override
                protected Map<String, Object> createTaskSettingsMap() {
                    return new HashMap<>();
                }

                @Override
                protected Map<String, Object> createSecretSettingsMap() {
                    return getSecretSettingsMap(SECRET);
                }

                @Override
                protected void assertModel(Model model, TaskType taskType, boolean modelIncludesSecrets) {
                    FireworksAiServiceTests.assertModel(model, modelIncludesSecrets);
                }

                @Override
                protected EnumSet<TaskType> supportedStreamingTasks() {
                    return EnumSet.noneOf(TaskType.class);
                }
            }
        ).enableUpdateModelTests(new UpdateModelConfiguration() {
            @Override
            protected Model createEmbeddingModel(@Nullable SimilarityMeasure similarityMeasure) {
                return createInternalEmbeddingModel(similarityMeasure, null);
            }
        }).build();
    }

    @Override
    public void testParseRequestConfig_CreatesACompletionModel() {
        // FireworksAI does not support the completion task type
    }

    @Override
    public void testUpdateModelWithEmbeddingDetails_NullSimilarityInOriginalModel() throws Exception {
        // FireworksAI defaults to COSINE similarity, not DOT_PRODUCT
        try (var service = testConfiguration.commonConfig().createService(threadPool, clientManager)) {
            var embeddingSize = randomNonNegativeInt();
            var model = testConfiguration.updateModelConfiguration().createEmbeddingModel(null);

            Model updatedModel = service.updateModelWithEmbeddingDetails(model, embeddingSize);

            assertEquals(SimilarityMeasure.COSINE, updatedModel.getServiceSettings().similarity());
            assertEquals(embeddingSize, updatedModel.getServiceSettings().dimensions().intValue());
        }
    }

    private static Map<String, Object> createServiceSettingsMap(TaskType taskType, ConfigurationParseContext parseContext) {
        var settingsMap = new HashMap<String, Object>(Map.of(ServiceFields.MODEL_ID, MODEL));

        if (taskType == TaskType.TEXT_EMBEDDING) {
            settingsMap.putAll(
                Map.of(ServiceFields.SIMILARITY, SIMILARITY.toString(), ServiceFields.DIMENSIONS, DIMENSIONS)
            );

            if (parseContext == ConfigurationParseContext.PERSISTENT) {
                settingsMap.put(FireworksAiEmbeddingsServiceSettings.DIMENSIONS_SET_BY_USER, true);
            }
        }

        return settingsMap;
    }

    private static void assertModel(Model model, boolean modelIncludesSecrets) {
        assertThat(model, instanceOf(FireworksAiEmbeddingsModel.class));

        var embeddingsModel = (FireworksAiEmbeddingsModel) model;
        assertThat(
            embeddingsModel.getServiceSettings(),
            is(
                new FireworksAiEmbeddingsServiceSettings(
                    MODEL,
                    URI.create(DEFAULT_URL),
                    SIMILARITY,
                    DIMENSIONS,
                    null,
                    true,
                    FireworksAiEmbeddingsServiceSettings.DEFAULT_RATE_LIMIT_SETTINGS
                )
            )
        );

        assertThat(embeddingsModel.getTaskSettings(), is(EmptyTaskSettings.INSTANCE));

        if (modelIncludesSecrets) {
            assertThat(embeddingsModel.getSecretSettings().apiKey().toString(), is(SECRET));
        } else {
            assertNull(embeddingsModel.getSecretSettings());
        }
    }

    private static FireworksAiEmbeddingsModel createInternalEmbeddingModel(
        @Nullable SimilarityMeasure similarityMeasure,
        @Nullable ChunkingSettings chunkingSettings
    ) {
        return new FireworksAiEmbeddingsModel(
            INFERENCE_ID,
            "service",
            new FireworksAiEmbeddingsServiceSettings(MODEL, URI.create(DEFAULT_URL), similarityMeasure, DIMENSIONS, null, false, null),
            chunkingSettings,
            new DefaultSecretSettings(new SecureString(SECRET.toCharArray()))
        );
    }

    private static FireworksAiService createService(ThreadPool threadPool, HttpClientManager clientManager) {
        var senderFactory = HttpRequestSenderTests.createSenderFactory(threadPool, clientManager);
        return new FireworksAiService(senderFactory, createWithEmptySettings(threadPool), mockClusterServiceEmpty());
    }
}
