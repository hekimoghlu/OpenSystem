/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 4, 2022.
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

#ifndef NON_PUBLIC_INHERITANCE_H
#define NON_PUBLIC_INHERITANCE_H

#define BLESS                                                                  \
  __attribute__((__language_attr__("private_fileid:main/blessed.code")))

const int PUBL_RETURN_VAL = 1;
const int PROT_RETURN_VAL = 2;
const int PRIV_RETURN_VAL = 3;

class BLESS Base {
public:
int publ(void) const { return PUBL_RETURN_VAL; }

protected:
int prot(void) const { return PROT_RETURN_VAL; }

private:
int priv(void) const { return PRIV_RETURN_VAL; }
};

class BLESS PublBase : public Base {};
class BLESS ProtBase : protected Base {};
class BLESS PrivBase : private Base {};

class BLESS PublPublBase : public PublBase {};
class BLESS ProtPublBase : protected PublBase {};
class BLESS PrivPublBase : private PublBase {};

class BLESS PublProtBase : public ProtBase {};
class BLESS ProtProtBase : protected ProtBase {};
class BLESS PrivProtBase : private ProtBase {};

class BLESS PublPrivBase : public PrivBase {};
class BLESS ProtPrivBase : protected PrivBase {};
class BLESS PrivPrivBase : private PrivBase {};

#endif /* NON_PUBLIC_INHERITANCE_H */
