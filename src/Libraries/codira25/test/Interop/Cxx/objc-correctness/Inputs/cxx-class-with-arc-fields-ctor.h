/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 16, 2024.
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

struct S {
  NSString *_Nullable A;
  NSString *_Nullable B;
  NSString *_Nullable C;

#ifdef S_NONTRIVIAL_DESTRUCTOR
  ~S() {}
#endif

  void dump() const {
    printf("%s\n", [A UTF8String]);
    printf("%s\n", [B UTF8String]);
    printf("%s\n", [C UTF8String]);
  }
};

inline void takeSFunc(S s) {
  s.dump();
}

struct NonTrivialLogDestructor {
    int x = 0;

    ~NonTrivialLogDestructor() {
        printf("~NonTrivialLogDestructor %d\n", x);
    }
};

@interface ClassWithNonTrivialDestructorIvar: NSObject

- (ClassWithNonTrivialDestructorIvar * _Nonnull)init;

- (void)takesS:(S)s;

@end

struct ReferenceStructToClassWithNonTrivialLogDestructorIvar {
    ClassWithNonTrivialDestructorIvar *_Nonnull x;

#ifdef S_NONTRIVIAL_DESTRUCTOR
    ~ReferenceStructToClassWithNonTrivialLogDestructorIvar() {}
#endif
};
