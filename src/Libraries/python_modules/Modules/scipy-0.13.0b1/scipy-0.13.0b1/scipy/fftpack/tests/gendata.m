/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 17, 2024.
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

x0 = linspace(0, 10, 11);
x1 = linspace(0, 10, 15);
x2 = linspace(0, 10, 16);
x3 = linspace(0, 10, 17);

x4 = randn(32, 1);
x5 = randn(64, 1);
x6 = randn(128, 1);
x7 = randn(256, 1);

y0 = dct(x0);
y1 = dct(x1);
y2 = dct(x2);
y3 = dct(x3);
y4 = dct(x4);
y5 = dct(x5);
y6 = dct(x6);
y7 = dct(x7);

save('test.mat', 'x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', ...
                 'y0', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7');
