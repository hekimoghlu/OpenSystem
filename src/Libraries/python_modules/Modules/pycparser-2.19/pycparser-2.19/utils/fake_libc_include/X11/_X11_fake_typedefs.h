/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 16, 2022.
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

#ifndef _X11_FAKE_TYPEDEFS_H
#define _X11_FAKE_TYPEDEFS_H

typedef char* XPointer;
typedef unsigned char KeyCode;
typedef unsigned int  CARD32;
typedef unsigned long VisualID;
typedef unsigned long XIMResetState;
typedef unsigned long XID;
typedef XID Window;
typedef XID Colormap;
typedef XID Cursor;
typedef XID Drawable;
typedef void* XtPointer;
typedef XtPointer XtRequestId;
typedef struct Display Display;
typedef struct Screen Screen;
typedef struct Status Status;
typedef struct Visual Visual;
typedef struct Widget *Widget;
typedef struct XColor XColor;
typedef struct XClassHint XClassHint;
typedef struct XEvent XEvent;
typedef struct XFontStruct XFontStruct;
typedef struct XGCValues XGCValues;
typedef struct XKeyEvent XKeyEvent;
typedef struct XKeyPressedEvent XKeyPressedEvent;
typedef struct XPoint XPoint;
typedef struct XRectangle XRectangle;
typedef struct XSelectionRequestEvent XSelectionRequestEvent;
typedef struct XWindowChanges XWindowChanges;
typedef struct _XGC _XCG;
typedef struct _XGC *GC;
typedef struct _XIC *XIC;
typedef struct _XIM *XIM;
typedef struct _XImage XImage;

#endif
