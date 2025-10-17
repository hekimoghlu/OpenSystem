/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 4, 2023.
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
#include <stdio.h>
#include <stdlib.h>

char *M, A, Z, E = 40, line[80], T[3];
int
main (int C)
{
  for (M = line + E, *line = A = scanf ("%d", &C); --E; line[E] = M[E] = E)
    printf ("._");
  for (; (A -= Z = !Z) || (printf ("\n|"), A = 39, C--); Z || printf (T))
    T[Z] = Z[A - (E = A[line - Z]) && !C
	     & A == M[A]
	     | RAND_MAX/3 < rand ()
	     || !C & !Z ? line[M[E] = M[A]] = E, line[M[A] = A - Z] =
	     A, "_." : " |"];
  return 0;
}
