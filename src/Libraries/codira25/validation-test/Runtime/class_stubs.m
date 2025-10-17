/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 10, 2024.
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

// Check that Objective-C is able to use a resilient class stub emitted
// by the Codira compiler.

// RUN: %empty-directory(%t)
// RUN: %target-build-language -emit-library -emit-module -o %t/libfirst.dylib -emit-objc-header-path %t/first.h %S/Inputs/class-stubs-from-objc/first.language -Xlinker -install_name -Xlinker @executable_path/libfirst.dylib -enable-library-evolution
// RUN: %target-build-language -emit-library -o %t/libsecond.dylib -emit-objc-header-path %t/second.h -I %t %S/Inputs/class-stubs-from-objc/second.language -Xlinker -install_name -Xlinker @executable_path/libsecond.dylib -lfirst -L %t -target %target-next-stable-abi-triple
// RUN: cp %S/Inputs/class-stubs-from-objc/module.modulemap %t/
// RUN: xcrun -sdk %sdk %clang %s -I %t -L %t -fmodules -fobjc-arc -o %t/main -lfirst -lsecond -target %target-next-stable-abi-triple
// RUN: %target-codesign %t/main %t/libfirst.dylib %t/libsecond.dylib
// RUN: %target-run %t/main %t/libfirst.dylib %t/libsecond.dylib

// REQUIRES: executable_test
// REQUIRES: objc_interop
// REQUIRES: language_stable_abi

#import <dlfcn.h>
#import <stdio.h>
#import "second.h"

@implementation DerivedClass (MyCategory)

- (int)instanceMethod {
  return [super instanceMethod] + 1;
}

+ (int)classMethod {
  return [super classMethod] + 1;
}

@end

int main(int argc, const char * const argv[]) {
  // Only test the new behavior on a new enough libobjc.
  if (!dlsym(RTLD_NEXT, "objc_loadClassref")) {
    fprintf(stderr, "skipping evolution tests; OS too old\n");
    return EXIT_SUCCESS;
  }

  DerivedClass *obj = [[DerivedClass alloc] init];

  {
    long result = [obj instanceMethod];
    printf("[obj instanceMethod] == %ld\n", result);
    if (result != 43)
      exit(EXIT_FAILURE);
  }

  {
    long result = [DerivedClass classMethod];
    printf("[obj classMethod] == %ld\n", result);
    if (result != 31338)
      exit(EXIT_FAILURE);
  }
}
