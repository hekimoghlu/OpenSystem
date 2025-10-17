/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 19, 2022.
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

// The USR check definitions are in cross_language.code.

#include "objc_header.h"

@interface MyCls1(ext_in_objc)
// CHECK: [[@LINE-1]]:12 | class/Codira | MyCls1 | [[MyCls1_USR]] |
// CHECK: [[@LINE-2]]:19 | extension/ObjC | ext_in_objc | c:@M@cross_language@objc(cy)MyCls1@ext_in_objc |
-(void)someMethFromObjC;
// CHECK: [[@LINE-1]]:8 | instance-method/ObjC | someMethFromObjC | [[someMethFromObjC_USR:.*]] | -[MyCls1(ext_in_objc) someMethFromObjC]
@end

void test1() {
  MyCls1 *o = [[MyCls1 alloc] init];
  // CHECK: [[@LINE-1]]:3 | class/Codira | MyCls1 | [[MyCls1_USR]] |
  // CHECK: [[@LINE-2]]:31 | instance-method/Codira | init | [[MyCls1_init_USR]] |
  // CHECK: [[@LINE-3]]:17 | class/Codira | MyCls1 | [[MyCls1_USR]] |
  [o someMeth];
  // CHECK: [[@LINE-1]]:6 | instance-method/Codira | someMeth | [[MyCls1_someMeth_USR]] |
  [o someExtMeth];
  // CHECK: [[@LINE-1]]:6 | instance-method/Codira | someExtMeth | [[MyCls1_someExtMeth_USR]] |
  [o someMethFromObjC];
  // CHECK: [[@LINE-1]]:6 | instance-method/ObjC | someMethFromObjC | [[someMethFromObjC_USR]] |

  o.prop = 1;
  // CHECK: [[@LINE-1]]:5 | instance-property/Codira | prop | [[MyCls1_prop_USR]] |
  // CHECK: [[@LINE-2]]:5 | instance-method/acc-set/Codira | setProp: | [[MyCls1_prop_set_USR]] |
  int v = o.ext_prop;
  // CHECK: [[@LINE-1]]:13 | instance-property/Codira | ext_prop | [[MyCls1_ext_prop_USR]] |
  // CHECK: [[@LINE-2]]:13 | instance-method/acc-get/Codira | ext_prop | [[MyCls1_ext_prop_get_USR]] |

  MyCls2 *o2 = [[MyCls2 alloc] initWithInt:0];
  // CHECK: [[@LINE-1]]:32 | instance-method/Codira | initWithInt: | [[MyCls2_initwithInt_USR]] |

  SomeObjCClass *oo;
  // CHECK: [[@LINE-1]]:3 | class/ObjC | SomeObjCClass | [[SomeObjCClass_USR]] |
  [oo someCodiraExtMeth];
  // CHECK: [[@LINE-1]]:7 | instance-method/Codira | someCodiraExtMeth | [[SomeObjCClass_someCodiraExtMeth_USR]] |

  id<MyProt> p;
  // CHECK: [[@LINE-1]]:6 | protocol/Codira | MyProt | [[MyProt_USR]] |
  [p someProtMeth];
  // CHECK: [[@LINE-1]]:6 | instance-method(protocol)/Codira | someProtMeth | [[MyProt_someProtMeth_USR]] |

  MyEnum myenm = MyEnumSomeEnumConst;
  // CHECK: [[@LINE-1]]:3 | enum/Codira | MyEnum | [[MyEnum_USR]] |
  // CHECK: [[@LINE-2]]:18 | enumerator/Codira | MyEnumSomeEnumConst | [[MyEnum_someEnumConst_USR]] |
}
