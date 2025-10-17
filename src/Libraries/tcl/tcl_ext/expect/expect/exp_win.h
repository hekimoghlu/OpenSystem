/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 13, 2024.
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
#include <tcl.h> /* For _ANSI_ARGS_ */

int exp_window_size_set   _ANSI_ARGS_ ((int fd));
int exp_window_size_get   _ANSI_ARGS_ ((int fd));

void  exp_win_rows_set    _ANSI_ARGS_ ((char* rows));
int   exp_win_rows_get    _ANSI_ARGS_ ((void));
void  exp_win_columns_set _ANSI_ARGS_ ((char* columns));
int   exp_win_columns_get _ANSI_ARGS_ ((void));

void  exp_win2_rows_set    _ANSI_ARGS_ ((int fd, char* rows));
int   exp_win2_rows_get    _ANSI_ARGS_ ((int fd));
void  exp_win2_columns_set _ANSI_ARGS_ ((int fd, char* columns));
int   exp_win2_columns_get _ANSI_ARGS_ ((int fd));
