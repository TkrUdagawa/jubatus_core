// Jubatus: Online machine learning framework for distributed environment
// Copyright (C) 2014 Preferred Networks and Nippon Telegraph and Telephone Corporation.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License version 2.1 as published by the Free Software Foundation.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

#include <gtest/gtest.h>

#include "compressive_storage.hpp"
#include "compressor.hpp"

using jubatus::util::lang::shared_ptr;

namespace jubatus {
namespace core {
namespace clustering {

class simple_compressor : public compressor::compressor {
 public:
  explicit simple_compressor()
      : compressor() {
  }

  void compress(
      const wplist& src,
      size_t bsize,
      size_t dstsize,
      wplist& dst) {
    dst.clear();
    for (size_t i = 0; i < dstsize && i < src.size(); ++i) {
      dst.push_back(src[src.size() - 1 - i]);
    }
  }
};

TEST(compressive_storage, carry_up) {
  int bucket_size = 2;
  int bucket_length = 2;
  int bicriteria_base_size = 1;
  int compressed_bucket_size = 1;
  double forgetting_factor = 1.0;
  double forgetting_threshold = 0.0;  // don't remove

  compressive_storage s("",
                        bucket_size,
                        bucket_length,
                        compressed_bucket_size,
                        bicriteria_base_size,
                        forgetting_factor,
                        forgetting_threshold);
  s.set_compressor(
       shared_ptr<compressor::compressor>(
           new simple_compressor()));

  weighted_point p;
  p.weight = 1.0;

  double decay = std::exp(-1);

  {
    // 1. { 0: [1] }
    // Get the first point into Lv0.
    s.add(p);
    wplist mine = s.get_mine();
    EXPECT_EQ(1u, mine.size());
    EXPECT_EQ(1.0, mine[0].weight);
  }

  {
    // 2. { 0: [1, 1] } -> { 1: [d] }
    // Lv0 coreset gets two points, then compress them and transfer it to Lv1.
    // Weights of points in Lv1 is `decay`.
    s.add(p);
    wplist mine = s.get_mine();
    EXPECT_EQ(1u, mine.size());
    EXPECT_EQ(decay, mine[0].weight);
  }

  {
    // 3. { 0: [1], 1: [d] }
    // Lv0 gets the next.
    s.add(p);
    wplist mine = s.get_mine();
    EXPECT_EQ(2u, mine.size());
    EXPECT_EQ(1.0, mine[0].weight);
    EXPECT_EQ(decay, mine[1].weight);
  }

  {
    // 4. { 0: [1, 1], 1: [d] } -> { 1: [d, d] } -> { 2: [d^2] }
    // Carry up twice. Then, Lv2 get a point whose weight is `decay * decay`.
    s.add(p);
    wplist mine = s.get_mine();
    EXPECT_EQ(1u, mine.size());
    EXPECT_DOUBLE_EQ(decay * decay, mine[0].weight);
  }
}

}  // namespace clustering
}  // namespace core
}  // namespace jubatus
