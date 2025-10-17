/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 8, 2025.
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
#if defined(WEBRTC_IOS)

#import <Foundation/Foundation.h>
#include <string.h>

#import "sdk/objc/helpers/NSString+StdString.h"

#include "absl/strings/string_view.h"
#include "rtc_base/checks.h"

namespace webrtc {
namespace test {

// For iOS, resource files are added to the application bundle in the root
// and not in separate folders as is the case for other platforms. This method
// therefore removes any prepended folders and uses only the actual file name.
std::string IOSResourcePath(absl::string_view name, absl::string_view extension) {
  @autoreleasepool {
    NSString* path = [NSString stringForAbslStringView:name];
    NSString* fileName = path.lastPathComponent;
    NSString* fileType = [NSString stringForAbslStringView:extension];
    // Get full pathname for the resource identified by the name and extension.
    NSString* pathString = [[NSBundle mainBundle] pathForResource:fileName
                                                           ofType:fileType];
    return [NSString stdStringForString:pathString];
  }
}

std::string IOSRootPath() {
  @autoreleasepool {
    NSBundle* mainBundle = [NSBundle mainBundle];
    return [NSString stdStringForString:mainBundle.bundlePath] + "/";
  }
}

// For iOS, we don't have access to the output directory. Return the path to the
// temporary directory instead. This is mostly used by tests that need to write
// output files to disk.
std::string IOSOutputPath()  {
  @autoreleasepool {
    NSString* tempDir = NSTemporaryDirectory();
    if (tempDir == nil)
        tempDir = @"/tmp";
    return [NSString stdStringForString:tempDir];
  }
}

}  // namespace test
}  // namespace webrtc

#endif  // defined(WEBRTC_IOS)
