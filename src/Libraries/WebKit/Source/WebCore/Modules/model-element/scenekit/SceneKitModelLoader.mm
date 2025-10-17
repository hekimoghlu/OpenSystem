/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 21, 2024.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#import "config.h"

#if HAVE(SCENEKIT)

#import "SceneKitModelLoader.h"

#import "MIMETypeRegistry.h"
#import "Model.h"
#import "ResourceError.h"
#import "SceneKitModelLoaderClient.h"
#import "SceneKitModelLoaderUSD.h"

namespace WebCore {

// Defining trivial virtual destructor in implementation to pin vtable.
SceneKitModelLoader::~SceneKitModelLoader() = default;

// No-op loader used for the cases where no loading actually happens (such as unsupported MIME type).
class SceneKitModelLoaderFailure final : public SceneKitModelLoader {
public:
    static Ref<SceneKitModelLoaderFailure> create(ResourceError error)
    {
        return adoptRef(*new SceneKitModelLoaderFailure(WTFMove(error)));
    }

    virtual ~SceneKitModelLoaderFailure() = default;
    virtual void cancel() override { }

    const ResourceError& error() const
    {
        return m_error;
    }

private:
    SceneKitModelLoaderFailure(ResourceError error)
        : m_error { WTFMove(error) }
    {
    }

    ResourceError m_error;
};


}

namespace WebCore {

static String mimeTypeUtilizingFileExtensionOverridingForLocalFiles(const Model& modelSource)
{
    if (modelSource.url().protocolIsFile() && (modelSource.mimeType().isEmpty() || modelSource.mimeType() == WebCore::defaultMIMEType())) {
        // FIXME: Getting the file extension from a URL seems like it should be in shared code.
        auto lastPathComponent = modelSource.url().lastPathComponent();
        auto position = lastPathComponent.reverseFind('.');
        if (position != WTF::notFound) {
            auto extension = lastPathComponent.substring(position + 1);

            return MIMETypeRegistry::mediaMIMETypeForExtension(extension);
        }
    }

    return modelSource.mimeType();
}

enum class ModelType {
    USD,
    Unknown
};

static ModelType modelType(Model& modelSource)
{
    auto mimeType = mimeTypeUtilizingFileExtensionOverridingForLocalFiles(modelSource);

    if (WebCore::MIMETypeRegistry::isUSDMIMEType(mimeType))
        return ModelType::USD;

    return ModelType::Unknown;
}

Ref<SceneKitModelLoader> loadSceneKitModel(Model& modelSource, SceneKitModelLoaderClient& client)
{
    switch (modelType(modelSource)) {
    case ModelType::USD:
        return loadSceneKitModelUsingUSDLoader(modelSource, client);
    case ModelType::Unknown:
        break;
    }

    auto loader = SceneKitModelLoaderFailure::create([NSError errorWithDomain:@"SceneKitModelLoader" code:-1 userInfo:@{
        NSLocalizedDescriptionKey: [NSString stringWithFormat:@"Unsupported MIME type: %s.", modelSource.mimeType().utf8().data()],
        NSURLErrorFailingURLErrorKey: (NSURL *)modelSource.url()
    }]);

    dispatch_async(dispatch_get_main_queue(), [weakClient = WeakPtr { client }, loader] {
        auto strongClient = weakClient.get();
        if (!strongClient)
            return;

        strongClient->didFailLoading(loader.get(), loader->error());
    });

    return loader;
}

}

#endif
