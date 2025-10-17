/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 17, 2022.
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
#ifndef ObjectWrapperInternal_h
#define ObjectWrapperInternal_h

/*
 * Type: ObjectWrapperRef
 *
 * Purpose:
 *  Provides a level of indirection between an object
 *  (e.g. IPConfigurationServiceRef) and any other object(s) that might need
 *  to reference it (e.g. SCDynamicStoreRef).
 *
 *  For an object that is invalidated by calling CFRelease
 *  (e.g. IPConfigurationServiceRef), that means there is normally only
 *  a single reference to that object. If there's an outstanding block that
 *  was scheduled (but not run) while calling CFRelease() on that object,
 *  when the block does eventually run, it can't validly reference
 *  that object anymore, it's been deallocated.
 *
 *  The ObjectWrapperRef is a simple, reference-counted structure that just
 *  stores an object pointer. In the *Deallocate function of the object,
 *  it synchronously calls ObjectWrapperClearObject(). When the block
 *  referencing the wrapper runs, it calls ObjectWrapperGetObject(), and if
 *  it is NULL, does not continue.
 */
typedef struct ObjectWrapper * ObjectWrapperRef;

ObjectWrapperRef
ObjectWrapperAlloc(const void * obj);

const void *
ObjectWrapperRetain(const void * info);

void
ObjectWrapperRelease(const void * info);

const void *
ObjectWrapperGetObject(ObjectWrapperRef wrapper);

void
ObjectWrapperClearObject(ObjectWrapperRef wrapper);

#endif /* ObjectWrapperInternal_h */
