/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 10, 2024.
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
#include "config.h"
#include "MediaEngineConfigurationFactory.h"

#include "MediaCapabilitiesDecodingInfo.h"
#include "MediaCapabilitiesEncodingInfo.h"
#include "MediaDecodingConfiguration.h"
#include "MediaEncodingConfiguration.h"
#include "MediaEngineConfigurationFactoryMock.h"
#include <wtf/Algorithms.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/Vector.h>

#if PLATFORM(COCOA)
#include "MediaEngineConfigurationFactoryCocoa.h"
#endif

#if USE(GSTREAMER)
#include "MediaEngineConfigurationFactoryGStreamer.h"
#endif

namespace WebCore {

static bool& mockEnabled()
{
    static bool enabled;
    return enabled;
}

using FactoryVector = Vector<MediaEngineConfigurationFactory::MediaEngineFactory>;
static FactoryVector defaultFactories()
{
    FactoryVector factories;
#if PLATFORM(COCOA)
    factories.append({ &createMediaPlayerDecodingConfigurationCocoa, nullptr });
#endif
#if USE(GSTREAMER)
    factories.append({ &createMediaPlayerDecodingConfigurationGStreamer, &createMediaPlayerEncodingConfigurationGStreamer });
#endif
    return factories;
}

static FactoryVector& factories()
{
    static NeverDestroyed factories = defaultFactories();
    return factories;
}

void MediaEngineConfigurationFactory::clearFactories()
{
    factories().clear();
}

void MediaEngineConfigurationFactory::resetFactories()
{
    factories() = defaultFactories();
}

void MediaEngineConfigurationFactory::installFactory(MediaEngineFactory&& factory)
{
    factories().append(WTFMove(factory));
}

bool MediaEngineConfigurationFactory::hasDecodingConfigurationFactory()
{
    return mockEnabled() || WTF::anyOf(factories(), [] (auto& factory) { return (bool)factory.createDecodingConfiguration; });
}

bool MediaEngineConfigurationFactory::hasEncodingConfigurationFactory()
{
    return mockEnabled() || WTF::anyOf(factories(), [] (auto& factory) { return (bool)factory.createEncodingConfiguration; });
}

void MediaEngineConfigurationFactory::createDecodingConfiguration(MediaDecodingConfiguration&& config, DecodingConfigurationCallback&& callback)
{
    if (mockEnabled()) {
        MediaEngineConfigurationFactoryMock::createDecodingConfiguration(WTFMove(config), WTFMove(callback));
        return;
    }

    auto factoryCallback = [] (auto factoryCallback, std::span<const MediaEngineFactory> nextFactories, MediaDecodingConfiguration&& config, DecodingConfigurationCallback&& callback) mutable {
        if (nextFactories.empty()) {
            callback({ { }, WTFMove(config) });
            return;
        }

        auto& factory = nextFactories[0];
        if (!factory.createDecodingConfiguration) {
            callback({ { }, WTFMove(config) });
            return;
        }

        factory.createDecodingConfiguration(WTFMove(config), [factoryCallback, nextFactories, config, callback = WTFMove(callback)] (MediaCapabilitiesDecodingInfo&& info) mutable {
            if (info.supported) {
                callback(WTFMove(info));
                return;
            }

            factoryCallback(factoryCallback, nextFactories.subspan(1), WTFMove(info.supportedConfiguration), WTFMove(callback));
        });
    };
    factoryCallback(factoryCallback, factories().span(), WTFMove(config), WTFMove(callback));
}

void MediaEngineConfigurationFactory::createEncodingConfiguration(MediaEncodingConfiguration&& config, EncodingConfigurationCallback&& callback)
{
    if (mockEnabled()) {
        MediaEngineConfigurationFactoryMock::createEncodingConfiguration(WTFMove(config), WTFMove(callback));
        return;
    }

    auto factoryCallback = [] (auto factoryCallback, std::span<const MediaEngineFactory> nextFactories, MediaEncodingConfiguration&& config, EncodingConfigurationCallback&& callback) mutable {
        if (nextFactories.empty()) {
            callback({ });
            return;
        }

        auto& factory = nextFactories[0];
        if (!factory.createEncodingConfiguration) {
            callback({ });
            return;
        }

        factory.createEncodingConfiguration(WTFMove(config), [factoryCallback, nextFactories, callback = WTFMove(callback)] (auto&& info) mutable {
            if (info.supported) {
                callback(WTFMove(info));
                return;
            }

            factoryCallback(factoryCallback, nextFactories.subspan(1), WTFMove(info.supportedConfiguration), WTFMove(callback));
        });
    };
    factoryCallback(factoryCallback, factories().span(), WTFMove(config), WTFMove(callback));
}

void MediaEngineConfigurationFactory::enableMock()
{
    mockEnabled() = true;
}

void MediaEngineConfigurationFactory::disableMock()
{
    mockEnabled() = false;
}

}
