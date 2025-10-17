/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 3, 2023.
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

// TEST_CONFIG OS=!exclavekit
// TEST_CFLAGS -framework Foundation
/*
TEST_RUN_OUTPUT
foo
bar
bar
foo
END
*/

// NOTE: This test won't catch problems when running against a root, so it's of
// limited utility, but it would at least catch things when testing against the
// shared cache.

#include <Foundation/Foundation.h>
#include <objc/runtime.h>

@interface NSBlock: NSObject @end

// NSBlock is a conveniently accessible superclass that (currently) has a constant cache.
@interface MyBlock: NSBlock
+(void)foo;
+(void)bar;
@end
@implementation MyBlock
+(void)foo {
  printf("foo\n");
}
+(void)bar {
  printf("bar\n");
}
@end

int main() {
  [MyBlock foo];
  [MyBlock bar];
  
  Method m1 = class_getClassMethod([MyBlock class], @selector(foo));
  Method m2 = class_getClassMethod([MyBlock class], @selector(bar));
  method_exchangeImplementations(m1, m2);
  
  [MyBlock foo];
  [MyBlock bar];
}
