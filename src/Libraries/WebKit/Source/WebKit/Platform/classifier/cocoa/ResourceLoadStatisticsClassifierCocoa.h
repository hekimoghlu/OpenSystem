/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 8, 2024.
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
#pragma once

#if HAVE(CORE_PREDICTION)

#include "ResourceLoadStatisticsClassifier.h"
#include <wtf/Platform.h>
#include <wtf/text/WTFString.h>

struct svm_model;

namespace WebKit {

class ResourceLoadStatisticsClassifierCocoa : public ResourceLoadStatisticsClassifier {
private:
    bool classify(unsigned, unsigned, unsigned) override;
    String storagePath();
    bool canUseCorePrediction();
    const struct svm_model* singletonPredictionModel();
    bool m_useCorePrediction { true };
    bool m_haveLoadedModel { false };
};
    
}

#endif
