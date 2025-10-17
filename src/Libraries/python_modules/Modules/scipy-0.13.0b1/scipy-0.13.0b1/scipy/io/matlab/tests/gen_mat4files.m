/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 17, 2022.
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

% Generates mat files for loadmat unit tests
% Uses save_matfile.m function
% This is the version for matlab 4

% work out matlab version and file suffix for test files
global FILEPREFIX FILESUFFIX
sepchar = '/';
if strcmp(computer, 'PCWIN'), sepchar = '\'; end
FILEPREFIX = [pwd sepchar 'data' sepchar];
mlv = version;
FILESUFFIX = ['_' mlv '_' computer '.mat'];

% basic double array
theta = 0:pi/4:2*pi;
save_matfile('testdouble', theta);

% string
save_matfile('teststring', '"Do nine men interpret?" "Nine men," I nod.')

% complex
save_matfile('testcomplex', cos(theta) + 1j*sin(theta));

% asymmetric array to check indexing
a = zeros(3, 5);
a(:,1) = [1:3]';
a(1,:) = 1:5;

% 2D matrix
save_matfile('testmatrix', a);

% minus number - tests signed int 
save_matfile('testminus', -1);

% single character
save_matfile('testonechar', 'r');

% string array
save_matfile('teststringarray', ['one  '; 'two  '; 'three']);

% sparse array
save_matfile('testsparse', sparse(a));

% sparse complex array
b = sparse(a);
b(1,1) = b(1,1) + j;
save_matfile('testsparsecomplex', b);

% Two variables in same file
save([FILEPREFIX 'testmulti' FILESUFFIX], 'a', 'theta')

