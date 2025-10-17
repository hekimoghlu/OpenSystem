/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 14, 2023.
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
#ifdef MP_8BIT
#define THE_MASK 127
#else
#define THE_MASK 32767
#endif

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mpi.c"

FILE *rng;

void rand_num(mp_int *a)
{
   int n, size;
   unsigned char buf[2048];

   size = 1 + ((fgetc(rng)<<8) + fgetc(rng)) % 101;
   buf[0] = (fgetc(rng)&1)?1:0;
   fread(buf+1, 1, size, rng);
   while (buf[1] == 0) buf[1] = fgetc(rng);
   mp_read_raw(a, buf, 1+size);
}

void rand_num2(mp_int *a)
{
   int n, size;
   unsigned char buf[2048];

   size = 10 + ((fgetc(rng)<<8) + fgetc(rng)) % 101;
   buf[0] = (fgetc(rng)&1)?1:0;
   fread(buf+1, 1, size, rng);
   while (buf[1] == 0) buf[1] = fgetc(rng);
   mp_read_raw(a, buf, 1+size);
}

#define mp_to64(a, b) mp_toradix(a, b, 64)

int main(void)
{
   int n, tmp;
   mp_int a, b, c, d, e;
   clock_t t1;
   char buf[4096];

   mp_init(&a);
   mp_init(&b);
   mp_init(&c);
   mp_init(&d);
   mp_init(&e);


   /* initial (2^n - 1)^2 testing, makes sure the comba multiplier works [it has the new carry code] */
/*
   mp_set(&a, 1);
   for (n = 1; n < 8192; n++) {
       mp_mul(&a, &a, &c);
       printf("mul\n");
       mp_to64(&a, buf);
       printf("%s\n%s\n", buf, buf);
       mp_to64(&c, buf);
       printf("%s\n", buf);

       mp_add_d(&a, 1, &a);
       mp_mul_2(&a, &a);
       mp_sub_d(&a, 1, &a);
   }
*/

   rng = fopen("/dev/urandom", "rb");
   if (rng == NULL) {
      rng = fopen("/dev/random", "rb");
      if (rng == NULL) {
         fprintf(stderr, "\nWarning:  stdin used as random source\n\n");
         rng = stdin;
      }
   }

   t1 = clock();
   for (;;) {
#if 0
      if (clock() - t1 > CLOCKS_PER_SEC) {
         sleep(2);
         t1 = clock();
      }
#endif
       n = fgetc(rng) % 15;

   if (n == 0) {
       /* add tests */
       rand_num(&a);
       rand_num(&b);
       mp_add(&a, &b, &c);
       printf("add\n");
       mp_to64(&a, buf);
       printf("%s\n", buf);
       mp_to64(&b, buf);
       printf("%s\n", buf);
       mp_to64(&c, buf);
       printf("%s\n", buf);
   } else if (n == 1) {
      /* sub tests */
       rand_num(&a);
       rand_num(&b);
       mp_sub(&a, &b, &c);
       printf("sub\n");
       mp_to64(&a, buf);
       printf("%s\n", buf);
       mp_to64(&b, buf);
       printf("%s\n", buf);
       mp_to64(&c, buf);
       printf("%s\n", buf);
   } else if (n == 2) {
       /* mul tests */
       rand_num(&a);
       rand_num(&b);
       mp_mul(&a, &b, &c);
       printf("mul\n");
       mp_to64(&a, buf);
       printf("%s\n", buf);
       mp_to64(&b, buf);
       printf("%s\n", buf);
       mp_to64(&c, buf);
       printf("%s\n", buf);
   } else if (n == 3) {
      /* div tests */
       rand_num(&a);
       rand_num(&b);
       mp_div(&a, &b, &c, &d);
       printf("div\n");
       mp_to64(&a, buf);
       printf("%s\n", buf);
       mp_to64(&b, buf);
       printf("%s\n", buf);
       mp_to64(&c, buf);
       printf("%s\n", buf);
       mp_to64(&d, buf);
       printf("%s\n", buf);
   } else if (n == 4) {
      /* sqr tests */
       rand_num(&a);
       mp_sqr(&a, &b);
       printf("sqr\n");
       mp_to64(&a, buf);
       printf("%s\n", buf);
       mp_to64(&b, buf);
       printf("%s\n", buf);
   } else if (n == 5) {
      /* mul_2d test */
      rand_num(&a);
      mp_copy(&a, &b);
      n = fgetc(rng) & 63;
      mp_mul_2d(&b, n, &b);
      mp_to64(&a, buf);
      printf("mul2d\n");
      printf("%s\n", buf);
      printf("%d\n", n);
      mp_to64(&b, buf);
      printf("%s\n", buf);
   } else if (n == 6) {
      /* div_2d test */
      rand_num(&a);
      mp_copy(&a, &b);
      n = fgetc(rng) & 63;
      mp_div_2d(&b, n, &b, NULL);
      mp_to64(&a, buf);
      printf("div2d\n");
      printf("%s\n", buf);
      printf("%d\n", n);
      mp_to64(&b, buf);
      printf("%s\n", buf);
   } else if (n == 7) {
      /* gcd test */
      rand_num(&a);
      rand_num(&b);
      a.sign = MP_ZPOS;
      b.sign = MP_ZPOS;
      mp_gcd(&a, &b, &c);
      printf("gcd\n");
      mp_to64(&a, buf);
      printf("%s\n", buf);
      mp_to64(&b, buf);
      printf("%s\n", buf);
      mp_to64(&c, buf);
      printf("%s\n", buf);
   } else if (n == 8) {
      /* lcm test */
      rand_num(&a);
      rand_num(&b);
      a.sign = MP_ZPOS;
      b.sign = MP_ZPOS;
      mp_lcm(&a, &b, &c);
      printf("lcm\n");
      mp_to64(&a, buf);
      printf("%s\n", buf);
      mp_to64(&b, buf);
      printf("%s\n", buf);
      mp_to64(&c, buf);
      printf("%s\n", buf);
   } else if (n == 9) {
      /* exptmod test */
      rand_num2(&a);
      rand_num2(&b);
      rand_num2(&c);
//      if (c.dp[0]&1) mp_add_d(&c, 1, &c);
      a.sign = b.sign = c.sign = 0;
      mp_exptmod(&a, &b, &c, &d);
      printf("expt\n");
      mp_to64(&a, buf);
      printf("%s\n", buf);
      mp_to64(&b, buf);
      printf("%s\n", buf);
      mp_to64(&c, buf);
      printf("%s\n", buf);
      mp_to64(&d, buf);
      printf("%s\n", buf);
   } else if (n == 10) {
      /* invmod test */
      rand_num2(&a);
      rand_num2(&b);
      b.sign = MP_ZPOS;
      a.sign = MP_ZPOS;
      mp_gcd(&a, &b, &c);
      if (mp_cmp_d(&c, 1) != 0) continue;
      if (mp_cmp_d(&b, 1) == 0) continue;
      mp_invmod(&a, &b, &c);
      printf("invmod\n");
      mp_to64(&a, buf);
      printf("%s\n", buf);
      mp_to64(&b, buf);
      printf("%s\n", buf);
      mp_to64(&c, buf);
      printf("%s\n", buf);
   } else if (n == 11) {
      rand_num(&a);
      mp_mul_2(&a, &a);
      mp_div_2(&a, &b);
      printf("div2\n");
      mp_to64(&a, buf);
      printf("%s\n", buf);
      mp_to64(&b, buf);
      printf("%s\n", buf);
   } else if (n == 12) {
      rand_num2(&a);
      mp_mul_2(&a, &b);
      printf("mul2\n");
      mp_to64(&a, buf);
      printf("%s\n", buf);
      mp_to64(&b, buf);
      printf("%s\n", buf);
   } else if (n == 13) {
      rand_num2(&a);
      tmp = abs(rand()) & THE_MASK;
      mp_add_d(&a, tmp, &b);
      printf("add_d\n");
      mp_to64(&a, buf);
      printf("%s\n%d\n", buf, tmp);
      mp_to64(&b, buf);
      printf("%s\n", buf);
   } else if (n == 14) {
      rand_num2(&a);
      tmp = abs(rand()) & THE_MASK;
      mp_sub_d(&a, tmp, &b);
      printf("sub_d\n");
      mp_to64(&a, buf);
      printf("%s\n%d\n", buf, tmp);
      mp_to64(&b, buf);
      printf("%s\n", buf);
   }
   }
   fclose(rng);
   return 0;
}

/* $Source: /cvs/libtom/libtommath/mtest/mtest.c,v $ */
/* $Revision: 1.2 $ */
/* $Date: 2005/05/05 14:38:47 $ */
