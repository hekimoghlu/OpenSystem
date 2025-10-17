/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 31, 2023.
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
#import "TextRecognitionResult.h"

#import "CharacterRange.h"
#import <wtf/RuntimeApplicationChecks.h>

#if USE(APPKIT)
#import <AppKit/NSAttributedString.h>
#else
#import <UIKit/NSAttributedString.h>
#endif

#import <pal/cocoa/VisionKitCoreSoftLink.h>

namespace WebCore {

#if ENABLE(IMAGE_ANALYSIS_ENHANCEMENTS)

RetainPtr<NSData> TextRecognitionResult::encodeVKCImageAnalysis(RetainPtr<VKCImageAnalysis> analysis)
{
    return [NSKeyedArchiver archivedDataWithRootObject:analysis.get() requiringSecureCoding:YES error:nil];
}

RetainPtr<VKCImageAnalysis> TextRecognitionResult::decodeVKCImageAnalysis(RetainPtr<NSData> data)
{
    if (!PAL::isVisionKitCoreFrameworkAvailable())
        return nil;

    // FIXME: This should use _enableStrictSecureDecodingMode or extract members into custom structures,
    // but that is blocked by rdar://108673895. In the meantime, just make sure this can't
    // be reached from outside the web content process to prevent sandbox escapes.
    RELEASE_ASSERT_WITH_SECURITY_IMPLICATION(isInWebProcess());

    return [NSKeyedUnarchiver unarchivedObjectOfClass:PAL::getVKCImageAnalysisClass() fromData:data.get() error:nil];
}

RetainPtr<NSAttributedString> stringForRange(const TextRecognitionResult& result, const CharacterRange& range)
{
    if (auto analysis = TextRecognitionResult::decodeVKCImageAnalysis(result.imageAnalysisData); [analysis respondsToSelector:@selector(_attributedStringForRange:)])
        return { [analysis _attributedStringForRange:static_cast<NSRange>(range)] };
    return nil;
}

#endif // ENABLE(IMAGE_ANALYSIS_ENHANCEMENTS)

} // namespace WebCore
