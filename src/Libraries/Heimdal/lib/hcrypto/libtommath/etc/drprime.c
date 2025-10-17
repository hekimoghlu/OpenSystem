/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 27, 2023.
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
#include <tommath.h>

int sizes[] = { 1+256/DIGIT_BIT, 1+512/DIGIT_BIT, 1+768/DIGIT_BIT, 1+1024/DIGIT_BIT, 1+2048/DIGIT_BIT, 1+4096/DIGIT_BIT };
int main(void)
{
   int res, x, y;
   char buf[4096];
   FILE *out;
   mp_int a, b;

   mp_init(&a);
   mp_init(&b);

   out = fopen("drprimes.txt", "w");
   for (x = 0; x < (int)(sizeof(sizes)/sizeof(sizes[0])); x++) {
   top:
       printf("Seeking a %d-bit safe prime\n", sizes[x] * DIGIT_BIT);
       mp_grow(&a, sizes[x]);
       mp_zero(&a);
       for (y = 1; y < sizes[x]; y++) {
           a.dp[y] = MP_MASK;
       }

       /* make a DR modulus */
       a.dp[0] = -1;
       a.used = sizes[x];

       /* now loop */
       res = 0;
       for (;;) {
          a.dp[0] += 4;
          if (a.dp[0] >= MP_MASK) break;
          mp_prime_is_prime(&a, 1, &res);
          if (res == 0) continue;
          printf("."); fflush(stdout);
          mp_sub_d(&a, 1, &b);
          mp_div_2(&b, &b);
          mp_prime_is_prime(&b, 3, &res);
          if (res == 0) continue;
          mp_prime_is_prime(&a, 3, &res);
          if (res == 1) break;
	}

        if (res != 1) {
           printf("Error not DR modulus\n"); sizes[x] += 1; goto top;
        } else {
           mp_toradix(&a, buf, 10);
           printf("\n\np == %s\n\n", buf);
           fprintf(out, "%d-bit prime:\np == %s\n\n", mp_count_bits(&a), buf); fflush(out);
        }
   }
   fclose(out);

   mp_clear(&a);
   mp_clear(&b);

   return 0;
}


/* $Source: /cvs/libtom/libtommath/etc/drprime.c,v $ */
/* $Revision: 1.2 $ */
/* $Date: 2005/05/05 14:38:47 $ */
