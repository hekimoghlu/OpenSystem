/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 30, 2023.
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
#import <Foundation/Foundation.h>

#ifdef WEBRTC_IOS_ENABLE_COVERAGE
extern "C" void __llvm_profile_set_filename(const char* name);
#endif

namespace rtc {
namespace test {

void ConfigureCoverageReportPath() {
#ifdef WEBRTC_IOS_ENABLE_COVERAGE
  static dispatch_once_t once_token;
  dispatch_once(&once_token, ^{
    // Writes the profraw file to the Documents directory, where the app has
    // write rights.
    NSArray* paths =
        NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString* documents_directory = [paths firstObject];
    NSString* file_name = [documents_directory stringByAppendingPathComponent:@"coverage.profraw"];

    // For documentation, see:
    // http://clang.llvm.org/docs/SourceBasedCodeCoverage.html
    __llvm_profile_set_filename([file_name cStringUsingEncoding:NSUTF8StringEncoding]);

    // Print the path for easier retrieval.
    NSLog(@"Coverage data at %@.", file_name);
  });
#endif  // ifdef WEBRTC_IOS_ENABLE_COVERAGE
}

}  // namespace test
}  // namespace rtc
