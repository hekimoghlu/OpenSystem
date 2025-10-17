/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 2, 2024.
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
#include "ResourceLoadStatisticsClassifierCocoa.h"

#if HAVE(CORE_PREDICTION)

#include "CorePredictionSPI.h"
#include "Logging.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/darwin/WeakLinking.h>

WTF_WEAK_LINK_FORCE_IMPORT(svm_load_model);

namespace WebKit {

bool ResourceLoadStatisticsClassifierCocoa::classify(unsigned subresourceUnderTopFrameDomainsCount, unsigned subresourceUniqueRedirectsToCount, unsigned subframeUnderTopFrameOriginsCount)
{
    if (!canUseCorePrediction())
        return classifyWithVectorThreshold(subresourceUnderTopFrameDomainsCount, subresourceUniqueRedirectsToCount, subframeUnderTopFrameOriginsCount);

    Vector<svm_node> features;

    if (subresourceUnderTopFrameDomainsCount)
        features.append({1, static_cast<double>(subresourceUnderTopFrameDomainsCount)});
    if (subresourceUniqueRedirectsToCount)
        features.append({2, static_cast<double>(subresourceUniqueRedirectsToCount)});
    if (subframeUnderTopFrameOriginsCount)
        features.append({3, static_cast<double>(subframeUnderTopFrameOriginsCount)});

    // Add termination node with index -1.
    features.append({-1, -1});

    double score;
    int classification = svm_predict_values(singletonPredictionModel(), features.data(), &score);
    LOG(ResourceLoadStatistics, "ResourceLoadStatisticsClassifierCocoa::classify(): Classified with CorePrediction.");
    return classification < 0;
}

String ResourceLoadStatisticsClassifierCocoa::storagePath()
{
    CFBundleRef webKitBundle = CFBundleGetBundleWithIdentifier(CFSTR("com.apple.WebKit"));
    RetainPtr<CFURLRef> resourceUrl = adoptCF(CFBundleCopyResourcesDirectoryURL(webKitBundle));
    resourceUrl = adoptCF(CFURLCreateCopyAppendingPathComponent(nullptr, resourceUrl.get(), CFSTR("corePrediction_model"), false));
    CFErrorRef error = nullptr;
    resourceUrl = adoptCF(CFURLCreateFilePathURL(nullptr, resourceUrl.get(), &error));

    if (error || !resourceUrl)
        return String();

    RetainPtr<CFStringRef> resourceUrlString = adoptCF(CFURLCopyFileSystemPath(resourceUrl.get(), kCFURLPOSIXPathStyle));
    return String(resourceUrlString.get());
}

bool ResourceLoadStatisticsClassifierCocoa::canUseCorePrediction()
{
    if (m_haveLoadedModel)
        return true;

    if (!m_useCorePrediction)
        return false;

    if (svm_load_model) {
        m_useCorePrediction = false;
        return false;
    }

    String storagePathStr = storagePath();
    if (storagePathStr.isNull() || storagePathStr.isEmpty()) {
        m_useCorePrediction = false;
        return false;
    }

    if (singletonPredictionModel()) {
        m_haveLoadedModel = true;
        return true;
    }

    m_useCorePrediction = false;
    return false;
}

const struct svm_model* ResourceLoadStatisticsClassifierCocoa::singletonPredictionModel()
{
    static std::optional<struct svm_model*> corePredictionModel;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        auto path = storagePath();
        if (path.isEmpty())
            return;

        corePredictionModel = svm_load_model(path.utf8().data());
    });

    if (corePredictionModel && corePredictionModel.value())
        return corePredictionModel.value();

    WTFLogAlways("ResourceLoadStatisticsClassifierCocoa::singletonPredictionModel(): Couldn't load model file at path %s.", storagePath().utf8().data());
    m_useCorePrediction = false;
    return nullptr;
}
}

#endif
