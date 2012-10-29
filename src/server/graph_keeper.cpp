// this program is automatically generated by jenerator. do not edit.
#include "../framework/keeper.hpp"
#include "../framework/aggregators.hpp"
#include "../common/exception.hpp"
#include "graph_types.hpp"
using namespace jubatus;
using namespace jubatus::framework;
int main(int args, char** argv){
  try{
    keeper k(keeper_argv(args,argv,"graph"));
    k.register_random<std::string >("create_node"); //pass nolock
    k.register_cht<2, int >("remove_node", pfi::lang::function<int(int,int)>(&pass<int >)); //update
    k.register_cht<2, int, property >("update_node", pfi::lang::function<int(int,int)>(&all_and)); //update
    k.register_cht<1, unsigned long, edge_info >("create_edge", pfi::lang::function<unsigned long(unsigned long,unsigned long)>(&all_and)); //nolock
    k.register_cht<2, int, unsigned long, edge_info >("update_edge", pfi::lang::function<int(int,int)>(&all_and)); //update
    k.register_cht<2, int, unsigned long >("remove_edge", pfi::lang::function<int(int,int)>(&all_and)); //update
    k.register_random<double, std::string, int, preset_query >("centrality"); //pass analysis
    k.register_broadcast<bool, preset_query >("add_centrality_query", pfi::lang::function<bool(bool,bool)>(&all_and)); //update
    k.register_broadcast<bool, preset_query >("add_shortest_path_query", pfi::lang::function<bool(bool,bool)>(&all_and)); //update
    k.register_broadcast<bool, preset_query >("remove_centrality_query", pfi::lang::function<bool(bool,bool)>(&all_and)); //update
    k.register_broadcast<bool, preset_query >("remove_shortest_path_query", pfi::lang::function<bool(bool,bool)>(&all_and)); //update
    k.register_random<std::vector<std::string >, shortest_path_req >("shortest_path"); //pass analysis
    k.register_broadcast<int >("update_index", pfi::lang::function<int(int,int)>(&all_and)); //update
    k.register_broadcast<int >("clear", pfi::lang::function<int(int,int)>(&all_and)); //update
    k.register_cht<2, node_info >("get_node", pfi::lang::function<node_info(node_info,node_info)>(&pass<node_info >)); //analysis
    k.register_cht<2, edge_info, unsigned long >("get_edge", pfi::lang::function<edge_info(edge_info,edge_info)>(&pass<edge_info >)); //analysis
    k.register_broadcast<bool, std::string >("save", pfi::lang::function<bool(bool,bool)>(&all_and)); //update
    k.register_broadcast<bool, std::string >("load", pfi::lang::function<bool(bool,bool)>(&all_and)); //update
    k.register_broadcast<std::map<std::string,std::map<std::string,std::string > > >("get_status", pfi::lang::function<std::map<std::string,std::map<std::string,std::string > >(std::map<std::string,std::map<std::string,std::string > >,std::map<std::string,std::map<std::string,std::string > >)>(&merge<std::string,std::map<std::string,std::string > >)); //analysis



    return k.run();
  } catch (const jubatus::exception::jubatus_exception& e) {
    std::cout << e.diagnostic_information(true) << std::endl;
    return -1;
  }
}
