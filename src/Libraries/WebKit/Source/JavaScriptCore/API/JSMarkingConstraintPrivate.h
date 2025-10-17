/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 1, 2023.
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
#ifndef JSMarkingConstraintPrivate_h
#define JSMarkingConstraintPrivate_h

#include <JavaScriptCore/JSContextRef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct JSMarker;
typedef struct JSMarker JSMarker;
typedef JSMarker *JSMarkerRef;

struct JSMarker {
    bool (*IsMarked)(JSMarkerRef, JSObjectRef);
    void (*Mark)(JSMarkerRef, JSObjectRef);
};

typedef void (*JSMarkingConstraint)(JSMarkerRef, void *userData);

JS_EXPORT void JSContextGroupAddMarkingConstraint(JSContextGroupRef, JSMarkingConstraint, void *userData);

#ifdef __cplusplus
}
#endif

#endif // JSMarkingConstraintPrivate_h

