/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 26, 2023.
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

#if TARGET_OS_IPHONE
#define RTC_PIXEL_FORMAT GL_LUMINANCE
#define SHADER_VERSION
#define VERTEX_SHADER_IN "attribute"
#define VERTEX_SHADER_OUT "varying"
#define FRAGMENT_SHADER_IN "varying"
#define FRAGMENT_SHADER_OUT
#define FRAGMENT_SHADER_COLOR "gl_FragColor"
#define FRAGMENT_SHADER_TEXTURE "texture2D"

@class EAGLContext;
typedef EAGLContext GlContextType;
#else
#define RTC_PIXEL_FORMAT GL_RED
#define SHADER_VERSION "#version 150\n"
#define VERTEX_SHADER_IN "in"
#define VERTEX_SHADER_OUT "out"
#define FRAGMENT_SHADER_IN "in"
#define FRAGMENT_SHADER_OUT "out vec4 fragColor;\n"
#define FRAGMENT_SHADER_COLOR "fragColor"
#define FRAGMENT_SHADER_TEXTURE "texture"

@class NSOpenGLContext;
typedef NSOpenGLContext GlContextType;
#endif
