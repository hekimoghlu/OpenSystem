/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 26, 2024.
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

#import "SceneKitModelLoaderUSD.h"

#import "Model.h"
#import "ResourceError.h"
#import "SceneKitModel.h"
#import "SceneKitModelLoader.h"
#import "SceneKitModelLoaderClient.h"
#import <pal/spi/cocoa/SceneKitSPI.h>

namespace WebCore {

class SceneKitModelLoaderUSD final : public SceneKitModelLoader {
public:
    static Ref<SceneKitModelLoaderUSD> create()
    {
        return adoptRef(*new SceneKitModelLoaderUSD());
    }

    virtual ~SceneKitModelLoaderUSD() = default;
    virtual void cancel() final { m_canceled = true; }

    bool isCanceled() const { return m_canceled; }

private:
    SceneKitModelLoaderUSD()
        : m_canceled { false }
    {
    }

    bool m_canceled;
};

class SceneKitModelUSD final : public SceneKitModel {
public:
    static Ref<SceneKitModelUSD> create(Ref<Model> modelSource, RetainPtr<SCNScene> scene)
    {
        return adoptRef(*new SceneKitModelUSD(WTFMove(modelSource), WTFMove(scene)));
    }

    virtual ~SceneKitModelUSD() = default;

private:
    SceneKitModelUSD(Ref<Model> modelSource, RetainPtr<SCNScene> scene)
        : m_modelSource { WTFMove(modelSource) }
        , m_scene { WTFMove(scene) }
    {
    }

    // SceneKitModel overrides.
    virtual const Model& modelSource() const override
    {
        return m_modelSource.get();
    }

    virtual SCNScene *defaultScene() const override
    {
        return m_scene.get();
    }

    virtual NSArray<SCNScene *> *scenes() const override
    {
        return @[ m_scene.get() ];
    }

    Ref<Model> m_modelSource;
    RetainPtr<SCNScene> m_scene;
};

static RetainPtr<NSURL> writeToTemporaryFile(WebCore::Model& modelSource)
{
    // FIXME: DO NOT SHIP!!! We must not write these to disk; we need SceneKit
    // to support reading USD files from its [SCNSceneSource initWithData:options:],
    // initializer but currently that does not work.

    auto [filePath, fileHandle] = FileSystem::openTemporaryFile("ModelFile"_s, ".usdz"_s);
    ASSERT(FileSystem::isHandleValid(fileHandle));

    size_t byteCount = FileSystem::writeToFile(fileHandle, modelSource.data()->makeContiguous()->span());
    ASSERT_UNUSED(byteCount, byteCount == modelSource.data()->size());
    FileSystem::closeFile(fileHandle);

    return adoptNS([[NSURL alloc] initFileURLWithPath:filePath]);
}

Ref<SceneKitModelLoader> loadSceneKitModelUsingUSDLoader(Model& modelSource, SceneKitModelLoaderClient& client)
{
    auto loader = SceneKitModelLoaderUSD::create();
    
    dispatch_async(dispatch_get_main_queue(), [weakClient = WeakPtr { client }, loader, modelSource = Ref { modelSource }] () mutable {
        // If the client has gone away, there is no reason to do any work.
        auto strongClient = weakClient.get();
        if (!strongClient)
            return;

        // If the caller has canceled the load, there is no reason to do any work.
        if (loader->isCanceled())
            return;

        auto url = writeToTemporaryFile(modelSource.get());

        auto source = adoptNS([[SCNSceneSource alloc] initWithURL:url.get() options:nil]);
        NSError *error = nil;
        RetainPtr scene = [source sceneWithOptions:@{ } error:&error];

        if (error) {
            strongClient->didFailLoading(loader.get(), ResourceError(error));
            [error release];
            return;
        }

        ASSERT(scene);

        strongClient->didFinishLoading(loader.get(), SceneKitModelUSD::create(WTFMove(modelSource), WTFMove(scene)));
    });

    return loader;
}

}

#endif
