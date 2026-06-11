// Debug probe: tower name index on updates + death behavior.
use source2_demo::prelude::*;
use std::io::BufReader;

#[derive(Default)]
struct Probe {
    printed: u32,
}

#[observer]
#[uses_entities]
#[uses_string_tables]
impl Probe {
    #[on_entity]
    fn on_entity(&mut self, ctx: &Context, event: EntityEvents, entity: &Entity) -> ObserverResult {
        let class = entity.class().name();
        if !matches!(class, "CDOTA_BaseNPC_Tower" | "CDOTA_BaseNPC_Fort") {
            return Ok(());
        }
        match event {
            EntityEvents::Updated => {
                let life: u8 = try_property!(entity, "m_lifeState").unwrap_or(99);
                if life != 0 {
                    let idx: Option<i32> = try_property!(entity, "m_pEntity.m_nameStringableIndex");
                    let name = idx.and_then(|i| {
                        ctx.string_tables().get_by_name("EntityNames").ok()
                            .and_then(|t| t.get_row(i as usize).ok().map(|r| r.key().to_string()))
                    });
                    let team: Option<u8> = try_property!(entity, "m_iTeamNum");
                    eprintln!("tick {} tower DIED life {} idx {:?} name {:?} team {:?}", ctx.tick(), life, idx, name, team);
                } else if self.printed < 5 {
                    self.printed += 1;
                    let idx: Option<i32> = try_property!(entity, "m_pEntity.m_nameStringableIndex");
                    eprintln!("tick {} tower update, idx {:?}", ctx.tick(), idx);
                }
            }
            EntityEvents::Deleted => {
                eprintln!("tick {} tower entity DELETED idx {}", ctx.tick(), entity.index());
            }
            _ => {}
        }
        Ok(())
    }
}

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("usage: probe <demofile>");
    let input = BufReader::new(std::fs::File::open(path)?);
    let mut parser = Parser::from_reader(input)?;
    parser.register_observer::<Probe>();
    parser.run_to_end()?;
    Ok(())
}
