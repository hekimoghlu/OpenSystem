/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 3, 2024.
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
#include "TelephoneNumberDetector.h"

#if ENABLE(TELEPHONE_NUMBER_DETECTION)

#include <wtf/SoftLinking.h>

#if USE(APPLE_INTERNAL_SDK)
#include <DataDetectorsCore/DDDFAScanner.h>
#else
typedef struct __DDDFAScanner DDDFAScanner, * DDDFAScannerRef;
struct __DDDFACache;
#endif

SOFT_LINK_PRIVATE_FRAMEWORK_OPTIONAL(DataDetectorsCore)
SOFT_LINK(DataDetectorsCore, DDDFACacheCreateFromFramework, struct __DDDFACache*, (), ())
SOFT_LINK(DataDetectorsCore, DDDFAScannerCreateFromCache, DDDFAScannerRef, (struct __DDDFACache* cache), (cache))
SOFT_LINK(DataDetectorsCore, DDDFAScannerFirstResultInUnicharArray, Boolean, (DDDFAScannerRef scanner, const UniChar* str, unsigned length, int* startPos, int* endPos), (scanner, str, length, startPos, endPos))

namespace WebCore {
namespace TelephoneNumberDetector {

static DDDFAScannerRef phoneNumbersScanner()
{
    static NeverDestroyed<RetainPtr<DDDFAScannerRef>> scanner;
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        if (DataDetectorsCoreLibrary()) {
            if (auto cache = adoptCF(DDDFACacheCreateFromFramework()))
                scanner.get() = adoptCF(DDDFAScannerCreateFromCache(cache.get()));
        }
    });
    return scanner.get().get();
}

void prewarm()
{
    // Prewarm on a background queue to avoid hanging the main thread.
    dispatch_async(dispatch_get_global_queue(0, 0), ^{
        phoneNumbersScanner();
    });
}

bool isSupported()
{
    return phoneNumbersScanner() != nullptr;
}

bool find(std::span<const UChar> buffer, int* startPos, int* endPos)
{
    ASSERT(isSupported());
    return DDDFAScannerFirstResultInUnicharArray(phoneNumbersScanner(), reinterpret_cast<const UniChar*>(buffer.data()), buffer.size(), startPos, endPos);
}

} // namespace TelephoneNumberDetector
} // namespace WebCore

#endif // ENABLE(TELEPHONE_NUMBER_DETECTION)
