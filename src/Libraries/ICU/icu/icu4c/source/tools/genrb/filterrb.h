/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 14, 2022.
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

// Â© 2018 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html

#ifndef __FILTERRB_H__
#define __FILTERRB_H__

#include <list>
#include <map>
#include <memory>
#include <ostream>
#include <string>

#include "unicode/utypes.h"


/**
 * Represents an absolute path into a resource bundle.
 * For example: "/units/length/meter"
 */
class ResKeyPath {
public:
    /** Constructs an empty path (top of tree) */
    ResKeyPath();

    /** Constructs from a string path */
    ResKeyPath(const std::string& path, UErrorCode& status);

    void push(const std::string& key);
    void pop();

    const std::list<std::string>& pieces() const;

  private:
    std::list<std::string> fPath;
};

std::ostream& operator<<(std::ostream& out, const ResKeyPath& value);


/**
 * Interface used to determine whether to include or reject pieces of a
 * resource bundle based on their absolute path.
 */
class PathFilter {
public:
    enum EInclusion {
        INCLUDE,
        PARTIAL,
        EXCLUDE
    };

    static const char* kEInclusionNames[];

    virtual ~PathFilter();

    /**
     * Returns an EInclusion on whether or not the given path should be included.
     *
     * INCLUDE = include the whole subtree
     * PARTIAL = recurse into the subtree
     * EXCLUDE = reject the whole subtree
     */
    virtual EInclusion match(const ResKeyPath& path) const = 0;
};


/**
 * Implementation of PathFilter for a list of inclusion/exclusion rules.
 *
 * The wildcard pattern "*" means that the subsequent filters are applied to
 * every other tree sharing the same parent.
 *
 * For example, given this list of filter rules:
 */
//     -/alabama
//     +/alabama/alaska/arizona
//     -/fornia/hawaii
//     -/mississippi
//     +/mississippi/michigan
//     +/mississippi/*/maine
//     -/mississippi/*/iowa
//     +/mississippi/louisiana/iowa
/*
 * You get the following structure:
 *
 *     SimpleRuleBasedPathFilter {
 *       included: PARTIAL
 *       alabama: {
 *         included: EXCLUDE
 *         alaska: {
 *           included: PARTIAL
 *           arizona: {
 *             included: INCLUDE
 *           }
 *         }
 *       }
 *       fornia: {
 *         included: PARTIAL
 *         hawaii: {
 *           included: EXCLUDE
 *         }
 *       }
 *       mississippi: {
 *         included: EXCLUDE
 *         louisiana: {
 *           included: PARTIAL
 *           iowa: {
 *             included: INCLUDE
 *           }
 *           maine: {
 *             included: INCLUDE
 *           }
 *         }
 *         michigan: {
 *           included: INCLUDE
 *           iowa: {
 *             included: EXCLUDE
 *           }
 *           maine: {
 *             included: INCLUDE
 *           }
 *         }
 *         * {
 *           included: PARTIAL
 *           iowa: {
 *             included: EXCLUDE
 *           }
 *           maine: {
 *             included: INCLUDE
 *           }
 *         }
 *       }
 *     }
 */
class SimpleRuleBasedPathFilter : public PathFilter {
public:
    void addRule(const std::string& ruleLine, UErrorCode& status);
    void addRule(const ResKeyPath& path, bool inclusionRule, UErrorCode& status);

    EInclusion match(const ResKeyPath& path) const override;

    void print(std::ostream& out) const;

private:
    struct Tree {

        Tree() = default;

        /** Copy constructor */
        Tree(const Tree& other);

        /**
         * Information on the USER-SPECIFIED inclusion/exclusion.
         *
         * INCLUDE = this path exactly matches a "+" rule
         * PARTIAL = this path does not match any rule, but subpaths exist
         * EXCLUDE = this path exactly matches a "-" rule
         */
        EInclusion fIncluded = PARTIAL;
        std::map<std::string, Tree> fChildren;
        std::unique_ptr<Tree> fWildcard;

        void applyRule(
            const ResKeyPath& path,
            std::list<std::string>::const_iterator it,
            bool inclusionRule,
            UErrorCode& status);

        bool isLeaf() const;

        void print(std::ostream& out, int32_t indent) const;
    };

    Tree fRoot;
};

std::ostream& operator<<(std::ostream& out, const SimpleRuleBasedPathFilter& value);


#endif //__FILTERRB_H__
