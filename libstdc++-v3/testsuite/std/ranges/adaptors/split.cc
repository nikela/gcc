// Copyright (C) 2020 Free Software Foundation, Inc.
//
// This file is part of the GNU ISO C++ Library.  This library is free
// software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the
// Free Software Foundation; either version 3, or (at your option)
// any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License along
// with this library; see the file COPYING3.  If not see
// <http://www.gnu.org/licenses/>.

// { dg-options "-std=gnu++2a" }
// { dg-do run { target c++2a } }

#include <algorithm>
#include <ranges>
#include <string>
#include <string_view>
#include <testsuite_hooks.h>
#include <testsuite_iterators.h>

using __gnu_test::test_range;
using __gnu_test::forward_iterator_wrapper;

namespace ranges = std::ranges;
namespace views = std::ranges::views;

using namespace std::literals;

void
test01()
{
  auto x = "the  quick  brown  fox"sv;
  auto p = std::string{"  "};
  auto v = x | views::split(p);
  auto i = v.begin();
  VERIFY( ranges::equal(*i++, "the"sv) );
  VERIFY( ranges::equal(*i++, "quick"sv) );
  VERIFY( ranges::equal(*i++, "brown"sv) );
  VERIFY( ranges::equal(*i++, "fox"sv) );
  VERIFY( i == v.end() );
}

void
test02()
{
  auto x = "the quick brown fox"sv;
  auto v = x | views::split(' ');
  auto i = v.begin();
  VERIFY( ranges::equal(*i++, "the"sv) );
  VERIFY( ranges::equal(*i++, "quick"sv) );
  VERIFY( ranges::equal(*i++, "brown"sv) );
  VERIFY( ranges::equal(*i++, "fox"sv) );
  VERIFY( i == v.end() );
}

void
test03()
{
  char x[] = "the quick brown fox";
  test_range<char, forward_iterator_wrapper> rx(x);
  auto v = rx | views::split(' ');
  auto i = v.begin();
  VERIFY( ranges::equal(*i++, "the"sv) );
  VERIFY( ranges::equal(*i++, "quick"sv) );
  VERIFY( ranges::equal(*i++, "brown"sv) );
  VERIFY( ranges::equal(*i++, "fox"sv) );
  VERIFY( i == v.end() );
}

int
main()
{
  test01();
  test02();
}
