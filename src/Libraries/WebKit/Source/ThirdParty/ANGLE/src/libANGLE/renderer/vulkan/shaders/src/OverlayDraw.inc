/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 11, 2024.
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

//
// Copyright 2022 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Limits:
// Note that max length and size is defined such that each buffer length is below 16KB, the minimum
// guaranteed supported size for uniform buffers.
#define MAX_TEXT_WIDGETS 32
#define MAX_GRAPH_WIDGETS 32

#define MAX_TEXT_LENGTH 256
#define MAX_GRAPH_SIZE 256

// Font information:
#define FONT_CHARACTERS 95

#extension GL_EXT_samplerless_texture_functions : require

struct TextWidgetData
{
    uvec4 coordinates;
    vec4 color;
    uvec4 fontSize;  // w unused.  xy has the font glyph width/height.  z has the mip.
    uvec4 text[MAX_TEXT_LENGTH / 16];
};

struct GraphWidgetData
{
    uvec4 coordinates;
    vec4 color;
    uvec4 valueWidth;  // yzw unused.  x should necessarily divide coordinate's z-x
    uvec4 values[MAX_GRAPH_SIZE / 4];
};

layout(set = 0, binding = 0) uniform TextWidgets
{
    TextWidgetData textWidgetsData[MAX_TEXT_WIDGETS];
};

layout(set = 0, binding = 1) uniform GraphWidgets
{
    GraphWidgetData graphWidgetsData[MAX_GRAPH_WIDGETS];
};

layout(set = 0, binding = 2) uniform texture2DArray font;

layout(push_constant) uniform PushConstants
{
    uvec2 viewportSize;
    // Whether drawing text or graph widgets
    bool isText;
    // Prerotation support
    bool rotateXY;
} params;
