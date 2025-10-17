/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 13, 2022.
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
#ifdef HAVE__ALLOCA
  EXPLICIT_SYMBOL(_alloca)
  EXPLICIT_SYMBOL2(alloca, _alloca)
#endif
#ifdef HAVE___ALLOCA
  EXPLICIT_SYMBOL(__alloca)
#endif
#ifdef HAVE___CHKSTK
  EXPLICIT_SYMBOL(__chkstk)
#endif
#ifdef HAVE____CHKSTK
  EXPLICIT_SYMBOL(___chkstk)
#endif
#ifdef HAVE___MAIN
  EXPLICIT_SYMBOL(__main) // FIXME: Don't call it.
#endif

#ifdef HAVE___ASHLDI3
  EXPLICIT_SYMBOL(__ashldi3)
#endif
#ifdef HAVE___ASHRDI3
  EXPLICIT_SYMBOL(__ashrdi3)
#endif
#ifdef HAVE___CMPDI2 // FIXME: unused
  EXPLICIT_SYMBOL(__cmpdi2)
#endif
#ifdef HAVE___DIVDI3
  EXPLICIT_SYMBOL(__divdi3)
#endif
#ifdef HAVE___FIXDFDI
  EXPLICIT_SYMBOL(__fixdfdi)
#endif
#ifdef HAVE___FIXSFDI
  EXPLICIT_SYMBOL(__fixsfdi)
#endif
#ifdef HAVE___FIXUNSDFDI
  EXPLICIT_SYMBOL(__fixunsdfdi)
#endif
#ifdef HAVE___FIXUNSSFDI
  EXPLICIT_SYMBOL(__fixunssfdi)
#endif
#ifdef HAVE___FLOATDIDF
  EXPLICIT_SYMBOL(__floatdidf)
#endif
#ifdef HAVE___FLOATDISF
  EXPLICIT_SYMBOL(__floatdisf)
#endif
#ifdef HAVE___LSHRDI3
  EXPLICIT_SYMBOL(__lshrdi3)
#endif
#ifdef HAVE___MODDI3
  EXPLICIT_SYMBOL(__moddi3)
#endif
#ifdef HAVE___UDIVDI3
  EXPLICIT_SYMBOL(__udivdi3)
#endif
#ifdef HAVE___UMODDI3
  EXPLICIT_SYMBOL(__umoddi3)
#endif

/* msvcrt */
#if defined(_MSC_VER)
  EXPLICIT_SYMBOL2(alloca, _alloca_probe)
#endif
