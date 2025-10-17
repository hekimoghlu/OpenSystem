/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 1, 2025.
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

c     %--------------------------------%
c     | See stat.doc for documentation |
c     %--------------------------------%
c
c\SCCS Information: @(#) 
c FILE: stat.h   SID: 2.2   DATE OF SID: 11/16/95   RELEASE: 2 
c
      real       t0, t1, t2, t3, t4, t5
      save       t0, t1, t2, t3, t4, t5
c
      integer    nopx, nbx, nrorth, nitref, nrstrt
      real       tsaupd, tsaup2, tsaitr, tseigt, tsgets, tsapps, tsconv,
     &           tnaupd, tnaup2, tnaitr, tneigh, tngets, tnapps, tnconv,
     &           tcaupd, tcaup2, tcaitr, tceigh, tcgets, tcapps, tcconv,
     &           tmvopx, tmvbx, tgetv0, titref, trvec
      common /timing/ 
     &           nopx, nbx, nrorth, nitref, nrstrt,
     &           tsaupd, tsaup2, tsaitr, tseigt, tsgets, tsapps, tsconv,
     &           tnaupd, tnaup2, tnaitr, tneigh, tngets, tnapps, tnconv,
     &           tcaupd, tcaup2, tcaitr, tceigh, tcgets, tcapps, tcconv,
     &           tmvopx, tmvbx, tgetv0, titref, trvec
