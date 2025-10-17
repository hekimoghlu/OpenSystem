/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 18, 2024.
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
/*
 *
 * (C) 2002,2005 by Marcin Dalecki <martin@dalecki.de>
 *
 * MARCIN DALECKI ASSUMES NO RESPONSIBILITY FOR THE USE OR INABILITY TO USE ANY
 * OF THIS SOFTWARE . THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY
 * KIND, AND MARCIN DALECKI EXPRESSLY DISCLAIMS ALL IMPLIED WARRANTIES,
 * INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef EnhancedBP_H
#define EnhancedBP_H

#include <Xm/PushBP.h>

#include "gui_xmebw.h"


/*
 * EnhancedButton class structure.
 */
typedef struct _XmEnhancedButtonClassPart
{
    Pixmap stipple_bitmap;
} XmEnhancedButtonClassPart;

/*
 * Full class record declaration for EnhancedButton class.
 */
typedef struct
{
    CoreClassPart core_class;
    XmPrimitiveClassPart primitive_class;
    XmLabelClassPart label_class;
    XmPushButtonClassPart pushbutton_class;
    XmEnhancedButtonClassPart enhancedbutton_class;
} XmEnhancedButtonClassRec;


extern XmEnhancedButtonClassRec xmEnhancedButtonClassRec;

/*
 * EnhancedButton instance record.
 */
typedef struct _XmEnhancedButtonPart
{
    // public resources
    String pixmap_data;
    String pixmap_file;
    Dimension spacing;
    int label_location;

    // private resources
    int pixmap_depth;
    Dimension pixmap_width;
    Dimension pixmap_height;
    Pixmap normal_pixmap;
    Pixmap armed_pixmap;
    Pixmap insensitive_pixmap;
    Pixmap highlight_pixmap;

    int doing_setvalues;
    int doing_destroy;
} XmEnhancedButtonPart;


/*
 * Full instance record declaration.
 */
typedef struct _XmEnhancedButtonRec
{
    CorePart core;
    XmPrimitivePart primitive;
    XmLabelPart label;
    XmPushButtonPart pushbutton;
    XmEnhancedButtonPart enhancedbutton;
} XmEnhancedButtonRec;

#endif
