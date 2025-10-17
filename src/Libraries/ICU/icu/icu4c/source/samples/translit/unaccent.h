/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 31, 2022.
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
#include "unicode/translit.h"
#include "unicode/normlzr.h"

using namespace icu;

class UnaccentTransliterator : public Transliterator {
    
 public:
    
    /**
     * Constructor
     */
    UnaccentTransliterator();

    /**
     * Destructor
     */
    virtual ~UnaccentTransliterator();

 protected:

    /**
     * Implement Transliterator API
     */
    void handleTransliterate(Replaceable& text,
                             UTransPosition& index,
                             UBool incremental) const override;

 private:

    /**
     * Unaccent a single character using normalizer.
     */
    char16_t unaccent(char16_t c) const;

    Normalizer normalizer;

public:

    /**
     * Return the class ID for this class.  This is useful only for
     * comparing to a return value from getDynamicClassID().  For example:
     * <pre>
     * .      Base* polymorphic_pointer = createPolymorphicObject();
     * .      if (polymorphic_pointer->getDynamicClassID() ==
     * .          Derived::getStaticClassID()) ...
     * </pre>
     * @return          The class ID for all objects of this class.
     * @stable ICU 2.0
     */
    static inline UClassID getStaticClassID() { return (UClassID)&fgClassID; };

    /**
     * Returns a unique class ID <b>polymorphically</b>.  This method
     * is to implement a simple version of RTTI, since not all C++
     * compilers support genuine RTTI.  Polymorphic operator==() and
     * clone() methods call this method.
     * 
     * <p>Concrete subclasses of Transliterator that wish clients to
     * be able to identify them should implement getDynamicClassID()
     * and also a static method and data member:
     * 
     * <pre>
     * static UClassID getStaticClassID() { return (UClassID)&fgClassID; }
     * static char fgClassID;
     * </pre>
     *
     * Subclasses that do not implement this method will have a
     * dynamic class ID of Transliterator::getStatisClassID().
     *
     * @return The class ID for this object. All objects of a given
     * class have the same class ID.  Objects of other classes have
     * different class IDs.
     * @stable ICU 2.0
     */
    UClassID getDynamicClassID() const override { return getStaticClassID(); };

private:

    /**
     * Class identifier for subclasses of Transliterator that do not
     * define their class (anonymous subclasses).
     */
    static const char fgClassID;
};
