/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 26, 2024.
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

#ifndef NON_PUBLIC_SHADOW_H
#define NON_PUBLIC_SHADOW_H

/// Used to distinguish which member we resolve to.
enum class Return {
  Publ_publOrPriv,
  Publ_publOrProt,
  Publ_publPrivShadowed,
  Publ_publPublShadowed,

  Prot_publOrProt,
  Prot_protOrPriv,
  Prot_protPrivShadowed,
  Prot_protPublShadowed,

  Priv_publOrPriv,
  Priv_protOrPriv,
  Priv_privPrivShadowed,
  Priv_privPublShadowed,

  Shadow_publPrivShadowed,
  Shadow_publPublShadowed,
  Shadow_protPrivShadowed,
  Shadow_protPublShadowed,
  Shadow_privPrivShadowed,
  Shadow_privPublShadowed,
};

struct Publ {
public:
  Return publOrPriv(void) const { return Return::Publ_publOrPriv; }
  Return publOrProt(void) const { return Return::Publ_publOrProt; }
  Return publPrivShadowed(void) const { return Return::Publ_publPrivShadowed; }
  Return publPublShadowed(void) const { return Return::Publ_publPublShadowed; }
};

struct Prot {
protected:
  Return publOrProt(void) const { return Return::Prot_publOrProt; }
  Return protOrPriv(void) const { return Return::Prot_protOrPriv; }
  Return protPrivShadowed(void) const { return Return::Prot_protPrivShadowed; }
  Return protPublShadowed(void) const { return Return::Prot_protPublShadowed; }
};

struct Priv {
private:
  Return publOrPriv(void) const { return Return::Priv_publOrPriv; }
  Return protOrPriv(void) const { return Return::Priv_protOrPriv; }
  Return privPrivShadowed(void) const { return Return::Priv_privPrivShadowed; }
  Return privPublShadowed(void) const { return Return::Priv_privPublShadowed; }
};

struct Shadow : Priv, Prot, Publ {
public:
  Return publPublShadowed(void) const {
    return Return::Shadow_publPublShadowed;
  }
  Return protPublShadowed(void) const {
    return Return::Shadow_protPublShadowed;
  }
  Return privPublShadowed(void) const {
    return Return::Shadow_privPublShadowed;
  }

private:
  Return publPrivShadowed(void) const {
    return Return::Shadow_publPrivShadowed;
  }
  Return protPrivShadowed(void) const {
    return Return::Shadow_protPrivShadowed;
  }
  Return privPrivShadowed(void) const {
    return Return::Shadow_privPrivShadowed;
  }
};

#endif /* NON_PUBLIC_SHADOW_H */
