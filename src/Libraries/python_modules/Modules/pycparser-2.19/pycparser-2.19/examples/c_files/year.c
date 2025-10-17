/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 12, 2024.
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
#include <string.h>
#include <stdlib.h>

void convert(int thousands, int hundreds, int tens, int ones)
{
char *num[] = {"", "One", "Two", "Three", "Four", "Five", "Six",
	       "Seven", "Eight", "Nine"};

char *for_ten[] = {"", "", "Twenty", "Thirty", "Fourty", "Fifty", "Sixty",
		   "Seventy", "Eighty", "Ninty"};

char *af_ten[] = {"Ten", "Eleven", "Twelve", "Thirteen", "Fourteen",
		  "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Ninteen"};

  printf("\nThe year in words is:\n");

  printf("%s thousand", num[thousands]);
  if (hundreds != 0)
    printf(" %s hundred", num[hundreds]);

  if (tens != 1)
    printf(" %s %s", for_ten[tens], num[ones]);
  else
    printf(" %s", af_ten[ones]);
}


int main()
{
int year;
int n1000, n100, n10, n1;

  printf("\nEnter the year (4 digits): ");
  scanf("%d", &year);

  if (year > 9999 || year < 1000)
  {
    printf("\nError !! The year must contain 4 digits.");
    exit(EXIT_FAILURE);
  }

  n1000 = year/1000;
  n100 = ((year)%1000)/100;
  n10 = (year%100)/10;
  n1 = ((year%10)%10);

  convert(n1000, n100, n10, n1);

return 0;
}


