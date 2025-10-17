/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 7, 2024.
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

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/ManagedStatic.h"
#include <cassert>
#include <xlocale.h>


namespace {
  struct locale_holder {
    locale_holder()
    : l(newlocale(LC_CTYPE_MASK,"en_US.UTF-8",LC_GLOBAL_LOCALE))
    {
      assert(NULL!=l);
    }
    ~locale_holder() {
      freelocale(l);
    }

    int mbswidth(llvm::SmallString<16> s) const {
       // this implementation assumes no '\0' in s
      assert(s.size()==strlen(s.c_str()));

      size_t size = mbstowcs_l(NULL,s.c_str(),0,l);
      assert(size!=(size_t)-1);
      if (size==0)
        return 0;
      llvm::SmallVector<wchar_t,200> ws(size);
      size = mbstowcs_l(&ws[0],s.c_str(),ws.size(),l);
      assert(ws.size()==size);
      return wcswidth_l(&ws[0],ws.size(),l);
    }

    int isprint(int c) const {
      return iswprint_l(c,l);
    }

  private:

    locale_t l;
  };

  llvm::ManagedStatic<locale_holder> l;
}

namespace llvm {
namespace sys {
namespace locale {

int columnWidth(StringRef s) {
  int width = l->mbswidth(s);
  assert(width>=0);
  return width;
}

bool isPrint(int c) {
  return l->isprint(c);
}

}
}
}
