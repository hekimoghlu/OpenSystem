/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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

@interface Super : TestRoot { 
  @public 
#if OLD
    // nothing
#else
    char superIvar;
#endif
}
@end


@interface ShrinkingSuper : TestRoot {
  @public
#if OLD
    id superIvar[5];
    __weak id superIvar2[5];
#else
    // nothing
#endif
}
@end;


@interface MoreStrongSuper : TestRoot {
  @public
#if OLD
    void *superIvar;
#else
    id superIvar;
#endif
}
@end;


@interface MoreWeakSuper : TestRoot {
  @public
#if OLD
    id superIvar;
#else
    __weak id superIvar;
#endif
}
@end;

@interface MoreWeak2Super : TestRoot {
  @public
#if OLD
    void *superIvar;
#else
    __weak id superIvar;
#endif
}
@end;

@interface LessStrongSuper : TestRoot {
  @public
#if OLD
    id superIvar;
#else
    void *superIvar;
#endif
}
@end;

@interface LessWeakSuper : TestRoot {
  @public
#if OLD
    __weak id superIvar;
#else
    id superIvar;
#endif
}
@end;

@interface LessWeak2Super : TestRoot {
  @public
#if OLD
    __weak id superIvar;
#else
    void *superIvar;
#endif
}
@end;

@interface NoGCChangeSuper : TestRoot {
  @public
    intptr_t d;
    char superc1;
#if OLD
    // nothing
#else
    char superc2;
#endif
}
@end

@interface RunsOf15 : TestRoot {
  @public
    id scan1;
    intptr_t skip15[15];
    id scan15[15];
    intptr_t skip15_2[15];
    id scan15_2[15];
#if OLD
    // nothing
#else
    intptr_t skip1;
#endif
}
@end
