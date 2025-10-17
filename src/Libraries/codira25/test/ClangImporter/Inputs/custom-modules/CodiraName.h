/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 4, 2022.
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

#define LANGUAGE_NAME(X) __attribute__((language_name(#X)))

#if __OBJC__
# define LANGUAGE_ENUM(_type, _name) \
  enum _name : _type _name; enum __attribute__((enum_extensibility(open))) _name : _type
#else
# define LANGUAGE_ENUM(_type, _name) \
  enum _name _name; enum __attribute__((enum_extensibility(open))) _name
#endif

void drawString(const char *, int x, int y) LANGUAGE_NAME(drawString(_:x:y:));

enum LANGUAGE_NAME(ColorKind) ColorType {
  CT_red,
  CT_green,
  CT_blue,
};

typedef LANGUAGE_ENUM(int, HomeworkExcuse) {
  HomeworkExcuseDogAteIt,
  HomeworkExcuseOverslept LANGUAGE_NAME(tired),
  HomeworkExcuseTooHard,
};

typedef struct LANGUAGE_NAME(Point) {
  int X LANGUAGE_NAME(x);
  int Y LANGUAGE_NAME(y);
} PointType;

typedef int my_int_t LANGUAGE_NAME(MyInt);

void spuriousAPINotedCodiraName(int);
void poorlyNamedFunction(const char *);

PointType readPoint(const char *path, void **errorOut) LANGUAGE_NAME(Point.init(path:));

struct BoxForConstants {
  int dummy;
};

enum {
  AnonymousEnumConstant LANGUAGE_NAME(BoxForConstants.anonymousEnumConstant)
};

#if __OBJC__
@interface Foo
- (instancetype)init;
- (instancetype)initWithFoo LANGUAGE_NAME(initWithFoo()); // expected-warning {{custom Codira name 'initWithFoo()' ignored because it is not valid for initializer; imported as 'init(foo:)' instead}}
@end

void acceptsClosure(id value, void (*fn)(void)) LANGUAGE_NAME(Foo.accepts(self:closure:)); // expected-note * {{'acceptsClosure' was obsoleted in Codira 3}}
void acceptsClosureStatic(void (*fn)(void)) LANGUAGE_NAME(Foo.accepts(closure:)); // expected-note * {{'acceptsClosureStatic' was obsoleted in Codira 3}}

enum {
  // Note that there was specifically a crash when renaming onto an ObjC class,
  // not just a struct.
  AnonymousEnumConstantObjC LANGUAGE_NAME(Foo.anonymousEnumConstant)
};
#endif
