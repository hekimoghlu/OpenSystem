/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 19, 2024.
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
#include "ProcessWarming.h"

#include "CommonAtomStrings.h"
#include "CommonVM.h"
#include "Font.h"
#include "FontCache.h"
#include "FontCascadeDescription.h"
#include "HTMLNames.h"
#include "MathMLNames.h"
#include "MediaQueryFeatures.h"
#include "QualifiedName.h"
#include "SVGNames.h"
#include "TagName.h"
#include "TelephoneNumberDetector.h"
#include "UserAgentStyle.h"
#include "WebKitFontFamilyNames.h"
#include "XLinkNames.h"
#include "XMLNSNames.h"
#include "XMLNames.h"

#if ENABLE(GPU_DRIVER_PREWARMING)
#include "GPUPrewarming.h"
#endif

namespace WebCore {

void ProcessWarming::initializeNames()
{
    initializeCommonAtomStrings();
    HTMLNames::init();
    QualifiedName::init();
    SVGNames::init();
    XLinkNames::init();
    MathMLNames::init();
    XMLNSNames::init();
    XMLNames::init();
    WebKitFontFamilyNames::init();
    initializeTagNameStrings();
}

void ProcessWarming::prewarmGlobally()
{
    initializeNames();
    
    // Prewarms user agent stylesheet.
    Style::UserAgentStyle::initDefaultStyleSheet();
    MQ::Features::allSchemas();
    
    // Prewarms JS VM.
    commonVM();

    // Prewarm font cache
    FontCache::prewarmGlobally();

#if ENABLE(TELEPHONE_NUMBER_DETECTION)
    TelephoneNumberDetector::prewarm();
#endif

#if ENABLE(GPU_DRIVER_PREWARMING)
    prewarmGPU();
#endif
}

WebCore::PrewarmInformation ProcessWarming::collectPrewarmInformation()
{
    return { FontCache::forCurrentThread().collectPrewarmInformation() };
}

void ProcessWarming::prewarmWithInformation(PrewarmInformation&& prewarmInfo)
{
    FontCache::forCurrentThread().prewarm(WTFMove(prewarmInfo.fontCache));
}

}
