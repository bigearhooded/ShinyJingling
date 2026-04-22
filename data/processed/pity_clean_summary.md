# Cleaned pity dataset summary

- Input rows: 618
- Dropped (LLM misparsed / ball not pollution): 2
- Reclassified single_pool->cross_pool (n>80): 14
- Kept rows: 616

## Scope counts (cleaned)
- `cross_pool_total`: 280
- `qualitative_no_n`: 70
- `single_pool`: 266

## single_pool subset stats (modeling target)
- count: 266
- with n: 266
- min/median/max: 1 / 3 / 80
- n=80 spike: 10
- n=1 spike: 97

## Drop log
- [llm_misparsed] n=5600 excerpt: 雪影我抓了两只异色基本都是5.60次污染出的 就一只爱分享 慈悲为怀我抓到现在 三个异色兔一个雪影一个拉特 见都没见过[捂脸]
- [llm_misparsed] n=1000 excerpt: 我刷两个鱼污染1000球两个异色双灯鱼还没出两只污染双灯鱼，一直褪色