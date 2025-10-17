/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 24, 2023.
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
typedef struct S {
  int x;
  int y;
  int z;
} S;

typedef struct T {
  S s;
} T;

int d(S *s) {
  ++s->x;
  s->x--;
  s->y = s->y + 1;
  int *c = &s->x;
  S ss;
  ss.x = 1;
  ss.x += 2;
  ss.z *= 2;
  return 0;
}
int b(S *s) {
  d(s);
  return 0;
}
int c(int x) {
  if (x) {
    c(x - 1);
  } else {
    S s;
    d(&s);
  }
  return 0;
}
int a(S *s) {
  b(s);
  c(1);
  return 0;
}
int e(void) {
  c(0);
  return 0;
}
int main(void) {
  int p = 3;
  S s;
  s.x = p + 1;
  s.y = 2;
  s.z = 3;
  a(&s);
  T t;
  t.s.x = 3;
}
