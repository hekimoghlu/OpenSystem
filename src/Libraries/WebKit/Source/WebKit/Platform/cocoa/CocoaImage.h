/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 1, 2022.
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
#import <CoreGraphics/CoreGraphics.h>
#import <wtf/Compiler.h>
#import <wtf/RetainPtr.h>

#if USE(APPKIT)
OBJC_CLASS NSImage;
using CocoaImage = NSImage;
#else
OBJC_CLASS UIImage;
using CocoaImage = UIImage;
#endif

OBJC_CLASS NSData;

namespace WebKit {

RetainPtr<NSData> transcode(CGImageRef, CFStringRef typeIdentifier);
std::pair<RetainPtr<NSData>, RetainPtr<CFStringRef>> transcodeWithPreferredMIMEType(CGImageRef, CFStringRef preferredMIMEType);

} // namespace WebKit
