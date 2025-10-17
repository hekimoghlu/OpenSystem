/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 28, 2024.
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

#include "ObjCWeak.h"
#include <objc/runtime.h>

extern id _Nullable 
objc_initWeak(id _Nullable * _Nonnull location, id _Nullable val);

void tryWeakReferencing(id (^makeThing)(void)) {
  id thingy;
  @autoreleasepool {
    thingy = [makeThing() retain];
  }
  
  id weakThingy = nil;
  objc_initWeak(&weakThingy, thingy);
  
  @autoreleasepool {
    fputs("before giving up strong reference:\n", stderr);
    id x = objc_loadWeak(&weakThingy);
    if (x) {
      fputs([[x description] UTF8String], stderr);
      fputs("\n", stderr);
    } else {
      fputs("Gone\n", stderr);
    }
  }
  
  [thingy release];
  thingy = nil;

  for (int i = 0; i < 100; i++) {
    @autoreleasepool {
      id tmp = makeThing();
      id weakTmp = nil;
      objc_initWeak(&weakTmp, tmp);
      objc_loadWeak(&weakTmp);
      objc_storeWeak(&weakTmp, nil);
    }
  }
   
  @autoreleasepool {
    fputs("after giving up strong reference:\n", stderr);
    id x = objc_loadWeak(&weakThingy);
    if (x) {
      fputs([[x description] UTF8String], stderr);
      fputs("\n", stderr);
    } else {
      fputs("Gone\n", stderr);
    }
  }
  objc_storeWeak(&weakThingy, nil);

  @autoreleasepool {
    fputs("after giving up weak reference:\n", stderr);
    id x = objc_loadWeak(&weakThingy);
    if (x) {
      fputs([[x description] UTF8String], stderr);
      fputs("\n", stderr);
    } else {
      fputs("Gone\n", stderr);
    }
  }
}
