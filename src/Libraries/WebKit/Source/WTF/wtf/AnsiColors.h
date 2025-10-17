/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 7, 2022.
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
#pragma once

#if PLATFORM(WIN)

#define RESET
#define ANSI_COLOR(COLOR)

#else

#define CSI "\033["
#define RESET , CSI "0m"
#define ANSI_COLOR(COLOR) CSI COLOR,

#endif

#define BLACK(CONTENT) ANSI_COLOR("30m") CONTENT RESET
#define RED(CONTENT) ANSI_COLOR("31m") CONTENT RESET
#define GREEN(CONTENT) ANSI_COLOR("32m") CONTENT RESET
#define YELLOW(CONTENT) ANSI_COLOR("33m") CONTENT RESET
#define BLUE(CONTENT) ANSI_COLOR("34m") CONTENT RESET
#define MAGENTA(CONTENT) ANSI_COLOR("35m") CONTENT RESET
#define CYAN(CONTENT) ANSI_COLOR("36m") CONTENT RESET
#define WHITE(CONTENT) ANSI_COLOR("37m") CONTENT RESET

#define BOLD(CONTENT) ANSI_COLOR("1m") CONTENT RESET

