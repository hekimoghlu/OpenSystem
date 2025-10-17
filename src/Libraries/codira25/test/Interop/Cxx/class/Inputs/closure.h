/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 20, 2023.
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

#ifndef __CLOSURE__
#define __CLOSURE__

struct NonTrivial {
  NonTrivial() noexcept { p = new int(123); }
  ~NonTrivial() { delete p; }
  NonTrivial(const NonTrivial &other) noexcept {
    p = new int(*other.p);
  }
  int *p;
};

void cfunc(void (^ _Nonnull block)(NonTrivial)) noexcept {
  block(NonTrivial());
}

void cfunc2(void (*_Nonnull fp)(NonTrivial)) noexcept { (*fp)(NonTrivial()); }

NonTrivial cfunc3(NonTrivial, int, NonTrivial);

#if __OBJC__
struct ARCStrong {
  id a;
};

void cfuncARCStrong(void (*_Nonnull)(ARCStrong)) noexcept ;
#endif

void cfuncReturnNonTrivial(NonTrivial (^_Nonnull)()) noexcept;
void cfuncReturnNonTrivial2(NonTrivial (*_Nonnull)()) noexcept;

struct ARCWeak {
#if __OBJC__
  __weak _Nullable id m;
#endif
};

void cfuncARCWeak(void (^ _Nonnull block)(ARCWeak)) noexcept {
  block(ARCWeak());
}

void cfunc(NonTrivial) noexcept;
void cfuncARCWeak(ARCWeak) noexcept;

void (* _Nonnull getFnPtr() noexcept)(NonTrivial) noexcept;
void (* _Nonnull getFnPtr2() noexcept)(ARCWeak) noexcept;

class SharedRef {
public:
  static SharedRef *_Nonnull makeSharedRef() { return new SharedRef(); }
  int _refCount = 1;

private:
  SharedRef() = default;

  SharedRef(const SharedRef &other) = delete;
  SharedRef &operator=(const SharedRef &other) = delete;
  SharedRef(SharedRef &&other) = delete;
  SharedRef &operator=(SharedRef &&other) = delete;
} __attribute__((language_attr("import_reference")))
__attribute__((language_attr("retain:retainSharedRef")))
__attribute__((language_attr("release:releaseSharedRef")));

inline void
cppGo(void (*_Nonnull takeConstSharedRef)(const SharedRef *_Nonnull x)) {
  SharedRef *ref = SharedRef::makeSharedRef();
  takeConstSharedRef(ref);
}

inline void retainSharedRef(SharedRef *_Nonnull x) { x->_refCount += 1; }
inline void releaseSharedRef(SharedRef *_Nonnull x) {
  x->_refCount -= 1;
  if (x->_refCount == 0) {
    delete x;
  }
}

#endif // __CLOSURE__
