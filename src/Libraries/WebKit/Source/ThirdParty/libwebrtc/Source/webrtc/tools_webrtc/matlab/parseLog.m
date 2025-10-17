/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 16, 2022.
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

function parsed = parseLog(filename)
%
% parsed = parseLog(filename)
% Parses a DataLog text file, with the filename specified in the string
% filename, into a struct with each column name as a field, and with the
% column data stored as a vector in that field.
%
% Arguments
%
% filename: A string with the name of the file to parse.
%
% Return value
%
% parsed: A struct containing each column parsed from the input file
%         as a field and with the column data stored as a vector in that 
%         field.
%

% Copyright (c) 2011 The WebRTC project authors. All Rights Reserved.
%
% Use of this source code is governed by a BSD-style license
% that can be found in the LICENSE file in the root of the source
% tree. An additional intellectual property rights grant can be found
% in the file PATENTS.  All contributing project authors may
% be found in the AUTHORS file in the root of the source tree.

table = importdata(filename, ',', 1);
if ~isstruct(table)
  error('Malformed file, possibly empty or lacking data entries')
end

colheaders = table.textdata;
if length(colheaders) == 1
  colheaders = regexp(table.textdata{1}, ',', 'split');
end

parsed = struct;
i = 1;
while i <= length(colheaders)
  % Checking for a multi-value column.
  m = regexp(colheaders{i}, '([\w\t]+)\[(\d+)\]', 'tokens');
  if ~isempty(m)
    % Parse a multi-value column
    n = str2double(m{1}{2}) - 1;
    parsed.(strrep(m{1}{1}, ' ', '_')) = table.data(:, i:i+n);
    i = i + n + 1;
  elseif ~isempty(colheaders{i})
    % Parse a single-value column
    parsed.(strrep(colheaders{i}, ' ', '_')) = table.data(:, i);
    i = i + 1;
  else
    error('Empty column');
  end
end
