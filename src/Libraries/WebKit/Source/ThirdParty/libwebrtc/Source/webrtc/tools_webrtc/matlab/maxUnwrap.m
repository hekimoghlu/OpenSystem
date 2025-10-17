/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 11, 2025.
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

function sequence = maxUnwrap(sequence, max)
%
% sequence = maxUnwrap(sequence, max)
% Unwraps when a wrap around is detected.
%
% Arguments
%
% sequence: The vector to unwrap.
% max: The maximum value that the sequence can take,
%      and after which it will wrap over to 0.
%
% Return value
%
% sequence: The unwrapped vector.
%

% Copyright (c) 2011 The WebRTC project authors. All Rights Reserved.
%
% Use of this source code is governed by a BSD-style license
% that can be found in the LICENSE file in the root of the source
% tree. An additional intellectual property rights grant can be found
% in the file PATENTS.  All contributing project authors may
% be found in the AUTHORS file in the root of the source tree.

sequence = round((unwrap(2 * pi * sequence / max) * max) / (2 * pi));
