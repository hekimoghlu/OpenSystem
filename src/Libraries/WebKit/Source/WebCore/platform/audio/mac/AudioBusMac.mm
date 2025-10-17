/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 27, 2025.
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

#if ENABLE(WEB_AUDIO)

#import "AudioBus.h"

#import "AudioFileReader.h"
#import <wtf/cocoa/SpanCocoa.h>

@interface WebCoreAudioBundleClass : NSObject
@end

@implementation WebCoreAudioBundleClass
@end

namespace WebCore {

RefPtr<AudioBus> AudioBus::loadPlatformResource(const char* name, float sampleRate)
{
    @autoreleasepool {
        NSBundle *bundle = [NSBundle bundleForClass:[WebCoreAudioBundleClass class]];
        NSURL *audioFileURL = [bundle URLForResource:[NSString stringWithUTF8String:name] withExtension:@"wav" subdirectory:@"audio"];
        if (NSData *audioData = [NSData dataWithContentsOfURL:audioFileURL options:NSDataReadingMappedIfSafe error:nil])
            return createBusFromInMemoryAudioFile(span(audioData), false, sampleRate);
    }

    ASSERT_NOT_REACHED();
    return nullptr;
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
