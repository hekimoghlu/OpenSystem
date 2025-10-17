/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 30, 2024.
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
#include "mk4.h"
#include "mk4str.h"

void QuickTest(int pos_, int len_)
{
  c4_ViewProp p1 ("_B");
  c4_IntProp p2 ("p2");
  
  c4_Storage s1;
  c4_View v1 = s1.GetAs("v1[_B[p2:I]]");

  int n = 0;
  static int sizes[] = {999, 999, 999, 3, 0};

  for (int i = 0; sizes[i]; ++i) {
    c4_View v;
    for (int j = 0; j < sizes[i]; ++j)
      v.Add(p2 [++n]);
    v1.Add(p1 [v]);
  }

  c4_View v2 = v1.Blocked();
  printf("%d\n", v2.GetSize());
    
  v2.RemoveAt(pos_, len_);
  printf("%d\n", v2.GetSize());

  puts("done");
}

int main(int argc, char** argv)
{
  QuickTest(999, 1200);
}
